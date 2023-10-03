/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            KU Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#include "fatrop_conic_interface.hpp"
#include <numeric>
#include <cstring>

#include <fatrop_conic_runtime_str.h>

namespace casadi {

  extern "C"
  int CASADI_CONIC_FATROP_EXPORT
  casadi_register_conic_fatrop(Conic::Plugin* plugin) {
    plugin->creator = FatropConicInterface::creator;
    plugin->name = "fatrop";
    plugin->doc = FatropConicInterface::meta_doc.c_str();
    plugin->version = CASADI_VERSION;
    plugin->options = &FatropConicInterface::options_;
    return 0;
  }

  extern "C"
  void CASADI_CONIC_FATROP_EXPORT casadi_load_conic_fatrop() {
    Conic::registerPlugin(casadi_register_conic_fatrop);
  }

  FatropConicInterface::FatropConicInterface(const std::string& name,
                                     const std::map<std::string, Sparsity>& st)
    : Conic(name, st) {
  }

  FatropConicInterface::~FatropConicInterface() {
    clear_mem();
  }

  const Options FatropConicInterface::options_
  = {{&Conic::options_},
     {{"N",
       {OT_INT,
        "OCP horizon"}},
      {"nx",
       {OT_INTVECTOR,
        "Number of states, length N+1"}},
      {"nu",
       {OT_INTVECTOR,
        "Number of controls, length N"}},
      {"ng",
       {OT_INTVECTOR,
        "Number of non-dynamic constraints, length N+1"}},
      {"fatrop",
       {OT_DICT,
        "Options to be passed to fatrop"}}}
  };

  void FatropConicInterface::init(const Dict& opts) {
    Conic::init(opts);

    casadi_int struct_cnt=0;

    // Read options
    for (auto&& op : opts) {
      if (op.first=="N") {
        N_ = op.second;
        struct_cnt++;
      } else if (op.first=="nx") {
        nxs_ = op.second;
        struct_cnt++;
      } else if (op.first=="nu") {
        nus_ = op.second;
        struct_cnt++;
      } else if (op.first=="ng") {
        ngs_ = op.second;
        struct_cnt++;
      }
    }

    bool detect_structure = struct_cnt==0;
    casadi_assert(struct_cnt==0 || struct_cnt==4,
      "You must either set all of N, nx, nu, ng; "
      "or set none at all (automatic detection).");

    const std::vector<int>& nx = nxs_;
    const std::vector<int>& ng = ngs_;
    const std::vector<int>& nu = nus_;

    Sparsity lamg_csp_, lam_ulsp_, lam_uusp_, lam_xlsp_, lam_xusp_, lam_clsp_;

    if (detect_structure) {
      /* General strategy: look for the xk+1 diagonal part in A
      */

      // Find the right-most column for each row in A -> A_skyline
      // Find the second-to-right-most column -> A_skyline2
      // Find the left-most column -> A_bottomline
      Sparsity AT = A_.T();
      std::vector<casadi_int> A_skyline;
      std::vector<casadi_int> A_skyline2;
      std::vector<casadi_int> A_bottomline;
      for (casadi_int i=0;i<AT.size2();++i) {
        casadi_int pivot = AT.colind()[i+1];
        A_bottomline.push_back(AT.row()[AT.colind()[i]]);
        if (pivot>AT.colind()[i]) {
          A_skyline.push_back(AT.row()[pivot-1]);
          if (pivot>AT.colind()[i]+1) {
            A_skyline2.push_back(AT.row()[pivot-2]);
          } else {
            A_skyline2.push_back(-1);
          }
        } else {
          A_skyline.push_back(-1);
          A_skyline2.push_back(-1);
        }
      }

      /*
      Loop over the right-most columns of A:
      they form the diagonal part due to xk+1 in gap constraints.
      detect when the diagonal pattern is broken -> new stage
      */
      casadi_int pivot = 0; // Current right-most element
      casadi_int start_pivot = pivot; // First right-most element that started the stage
      casadi_int cg = 0; // Counter for non-gap-closing constraints
      for (casadi_int i=0;i<na_;++i) { // Loop over all rows
        bool commit = false; // Set true to jump to the stage
        if (A_skyline[i]>pivot+1) { // Jump to a diagonal in the future
          nus_.push_back(A_skyline[i]-pivot-1); // Size of jump equals number of states
          commit = true;
        } else if (A_skyline[i]==pivot+1) { // Walking the diagonal
          if (A_skyline2[i]<start_pivot) { // Free of below-diagonal entries?
            pivot++;
          } else {
            nus_.push_back(0); // We cannot but conclude that we arrived at a new stage
            commit = true;
          }
        } else { // non-gap-closing constraint detected
          cg++;
        }

        if (commit) {
          nxs_.push_back(pivot-start_pivot+1);
          ngs_.push_back(cg); cg=0;
          start_pivot = A_skyline[i];
          pivot = A_skyline[i];
        }
      }
      nxs_.push_back(pivot-start_pivot+1);

      // Correction for k==0
      nxs_[0] = A_skyline[0];
      nus_[0] = 0;
      ngs_.erase(ngs_.begin());
      casadi_int cN=0;
      for (casadi_int i=na_-1;i>=0;--i) {
        if (A_bottomline[i]<start_pivot) break;
        cN++;
      }
      ngs_.push_back(cg-cN);
      ngs_.push_back(cN);

      N_ = nus_.size();
      if (verbose_) {
        casadi_message("Detected structure: N " + str(N_) + ", nx " + str(nx) + ", "
          "nu " + str(nu) + ", ng " + str(ng) + ".");
      }
      nus_.push_back(0);
    }

    uout() << "nx,nu,ng" << nx << nu << ng << std::endl;

    casadi_assert_dev(nx.size()==N_+1);
    casadi_assert_dev(nu.size()==N_+1);
    casadi_assert_dev(ng.size()==N_+1);

    casadi_assert(nx_ == std::accumulate(nx.begin(), nx.end(), 0) + // NOLINT
      std::accumulate(nu.begin(), nu.end(), 0),
      "sum(nx)+sum(nu) = must equal total size of variables (" + str(nx_) + "). "
      "Structure is: N " + str(N_) + ", nx " + str(nx) + ", "
      "nu " + str(nu) + ", ng " + str(ng) + ".");
    casadi_assert(na_ == std::accumulate(nx.begin()+1, nx.end(), 0) + // NOLINT
      std::accumulate(ng.begin(), ng.end(), 0),
      "sum(nx+1)+sum(ng) = must equal total size of constraints (" + str(na_) + "). "
      "Structure is: N " + str(N_) + ", nx " + str(nx) + ", "
      "nu " + str(nu) + ", ng " + str(ng) + ".");
    // Load library HPIPM when applicable
    std::string searchpath;

    /* Disassemble A input into:
       A B I
       C D
           A B I
           C D
    */
    casadi_int offset_r = 0, offset_c = 0;
    for (casadi_int k=0;k<N_;++k) { // Loop over blocks
      A_blocks.push_back({offset_r,        offset_c,            nx[k+1], nx[k]});
      B_blocks.push_back({offset_r,        offset_c+nx[k],      nx[k+1], nu[k]});
      C_blocks.push_back({offset_r+nx[k+1], offset_c,           ng[k], nx[k]});
      D_blocks.push_back({offset_r+nx[k+1], offset_c+nx[k],     ng[k], nu[k]});

      offset_c+= nx[k]+nu[k];
      if (k+1<N_)
        I_blocks.push_back({offset_r, offset_c, nx[k+1], nx[k+1]});
      else
        I_blocks.push_back({offset_r, offset_c, nx[k+1], nx[k+1]});
      offset_r+= nx[k+1]+ng[k];
    }

    C_blocks.push_back({offset_r, offset_c,            ng[N_], nx[N_]});
    D_blocks.push_back({offset_r, offset_c+nx[N_],     ng[N_], nu[N_]});

    Asp_ = blocksparsity(na_, nx_, A_blocks);
    Bsp_ = blocksparsity(na_, nx_, B_blocks);
    Csp_ = blocksparsity(na_, nx_, C_blocks);
    Dsp_ = blocksparsity(na_, nx_, D_blocks);
    Isp_ = blocksparsity(na_, nx_, I_blocks, true);

    Sparsity total = Asp_ + Bsp_ + Csp_ + Dsp_ + Isp_;

    casadi_assert((A_ + total).nnz() == total.nnz(),
      "HPIPM: specified structure of A does not correspond to what the interface can handle. "
      "Structure is: N " + str(N_) + ", nx " + str(nx) + ", nu " + str(nu) + ", "
      "ng " + str(ng) + ".");
    casadi_assert_dev(total.nnz() == Asp_.nnz() + Bsp_.nnz() + Csp_.nnz() + Dsp_.nnz()
                      + Isp_.nnz());

    /* Disassemble H input into:
       Q S'
       S R
           Q S'
           S R

       Multiply by 2
    */
    casadi_int offset = 0;
    for (casadi_int k=0;k<N_+1;++k) { // Loop over blocks
      R_blocks.push_back({offset+nx[k], offset+nx[k],       nu[k], nu[k]});
      S_blocks.push_back({offset+nx[k], offset,             nu[k], nx[k]});
      Q_blocks.push_back({offset,       offset,             nx[k], nx[k]});
      offset+= nx[k]+nu[k];
    }

    Rsp_ = blocksparsity(nx_, nx_, R_blocks);
    Ssp_ = blocksparsity(nx_, nx_, S_blocks);
    Qsp_ = blocksparsity(nx_, nx_, Q_blocks);

    total = Rsp_ + Ssp_ + Qsp_ + Ssp_.T();
    casadi_assert((H_ + total).nnz() == total.nnz(),
      "HPIPM: specified structure of H does not correspond to what the interface can handle. "
      "Structure is: N " + str(N_) + ", nx " + str(nx) + ", nu " + str(nu) + ", "
      "ng " + str(ng) + ".");
    casadi_assert_dev(total.nnz() == Rsp_.nnz() + 2*Ssp_.nnz() + Qsp_.nnz());

    /* Disassemble LBA/UBA input into:
       b
       lg/ug

       b
       lg/ug
    */
    offset = 0;

    for (casadi_int k=0;k<N_;++k) {
      b_blocks.push_back({offset,   0, nx[k+1], 1}); offset+= nx[k+1];
      lug_blocks.push_back({offset, 0, ng[k], 1}); offset+= ng[k];
    }
    lug_blocks.push_back({offset, 0, ng[N_], 1});

    bsp_ = blocksparsity(na_, 1, b_blocks);
    lugsp_ = blocksparsity(na_, 1, lug_blocks);
    total = bsp_ + lugsp_;
    casadi_assert_dev(total.nnz() == bsp_.nnz() + lugsp_.nnz());
    casadi_assert_dev(total.nnz() == na_);

    /* Disassemble G/X0 input into:
       r/u
       q/x

       r/u
       q/x
    */
    offset = 0;

    for (casadi_int k=0;k<N_+1;++k) {
      x_blocks.push_back({offset, 0, nx[k], 1}); offset+= nx[k];
      u_blocks.push_back({offset, 0, nu[k], 1}); offset+= nu[k];
    }

    usp_ = blocksparsity(nx_, 1, u_blocks);
    xsp_ = blocksparsity(nx_, 1, x_blocks);
    total = usp_ + xsp_;
    casadi_assert_dev(total.nnz() == usp_.nnz() + xsp_.nnz());
    casadi_assert_dev(total.nnz() == nx_);

    std::vector< casadi_ocp_block > theirs_u_blocks, theirs_x_blocks;
    offset = 0;

    for (casadi_int k=0;k<N_;++k) {
      theirs_u_blocks.push_back({offset, 0, nu[k], 1}); offset+= nu[k];
      theirs_x_blocks.push_back({offset, 0, nx[k], 1}); offset+= nx[k];
    }
    theirs_x_blocks.push_back({offset, 0, nx[N_], 1});

    theirs_usp_ = blocksparsity(nx_, 1, theirs_u_blocks);
    theirs_xsp_ = blocksparsity(nx_, 1, theirs_x_blocks);
    total = theirs_usp_ + theirs_xsp_;
    casadi_assert_dev(total.nnz() == theirs_usp_.nnz() + theirs_xsp_.nnz());
    casadi_assert_dev(total.nnz() == nx_);

    offset = 0;
    std::vector< casadi_ocp_block > lamg_gap_blocks;
    for (casadi_int k=0;k<N_;++k) {
      lamg_gap_blocks.push_back({offset,       0, nx[k+1], 1});offset+= nx[k+1] + ng[k];
    }
    lamg_gapsp_ = blocksparsity(na_, 1, lamg_gap_blocks);
    lamg_csp_ = lamg_gapsp_.pattern_inverse();

    offset = 0;

    for (casadi_int k=0;k<N_;++k) {
      lam_ul_blocks.push_back({offset, 0, nu[k], 1}); offset+= nu[k];
      lam_xl_blocks.push_back({offset, 0, nx[k], 1}); offset+= nx[k];
      lam_uu_blocks.push_back({offset, 0, nu[k], 1}); offset+= nu[k];
      lam_xu_blocks.push_back({offset, 0, nx[k], 1}); offset+= nx[k];
      lam_cl_blocks.push_back({offset, 0, ng[k], 1}); offset+= ng[k];
      lam_cu_blocks.push_back({offset, 0, ng[k], 1}); offset+= ng[k];
    }
    lam_xl_blocks.push_back({offset, 0, nx[N_], 1}); offset+= nx[N_];
    lam_xu_blocks.push_back({offset, 0, nx[N_], 1}); offset+= nx[N_];
    lam_cl_blocks.push_back({offset, 0, ng[N_], 1}); offset+= ng[N_];
    lam_cu_blocks.push_back({offset, 0, ng[N_], 1}); offset+= ng[N_];

    lam_ulsp_ = blocksparsity(offset, 1, lam_ul_blocks);
    lam_uusp_ = blocksparsity(offset, 1, lam_uu_blocks);
    lam_xlsp_ = blocksparsity(offset, 1, lam_xl_blocks);
    lam_xusp_ = blocksparsity(offset, 1, lam_xu_blocks);
    lam_clsp_ = blocksparsity(offset, 1, lam_cl_blocks);
    lam_cusp_ = blocksparsity(offset, 1, lam_cu_blocks);

    pisp_ = Sparsity::dense(std::accumulate(nx.begin()+1, nx.end(), 0), 1);  // NOLINT

    total = lam_ulsp_ + lam_uusp_ + lam_xlsp_ + lam_xusp_ + lam_clsp_ + lam_cusp_;
    casadi_assert_dev(total.nnz() == lam_ulsp_.nnz() + lam_uusp_.nnz() + lam_xlsp_.nnz() +
      lam_xusp_.nnz() + lam_clsp_.nnz() + lam_cusp_.nnz());
    casadi_assert_dev(total.nnz() == offset);

    theirs_Xsp_ = Sparsity::dense(std::accumulate(nx.begin(), nx.end(), 0), 1);  // NOLINT
    theirs_Usp_ = Sparsity::dense(std::accumulate(nu.begin(), nu.end(), 0), 1);  // NOLINT

    nus_.push_back(0);
    zeros_.resize(N_+1, 0);

    uout() << "nus" << nus_ << std::endl;

    set_fatrop_conic_prob();

    // Allocate memory
    casadi_int sz_arg, sz_res, sz_w, sz_iw;
    casadi_fatrop_work(&p_, &sz_arg, &sz_res, &sz_iw, &sz_w);

    uout() << sz_arg << std::endl;
    uout() << sz_res << std::endl;
    uout() << sz_iw << std::endl;
    uout() << sz_w << std::endl;

    alloc_arg(sz_arg, true);
    alloc_res(sz_res, true);
    alloc_iw(sz_iw, true);
    alloc_w(sz_w, true);



  }

  std::vector<casadi_int> fatrop_blocks_pack(const std::vector<casadi_ocp_block>& blocks) {
    size_t N = blocks.size();
    std::vector<casadi_int> ret(4*N);
    casadi_int* r = get_ptr(ret);
    for (casadi_int i=0;i<N;++i) {
      *r++ = blocks[i].offset_r;
      *r++ = blocks[i].offset_c;
      *r++ = blocks[i].rows;
      *r++ = blocks[i].cols;
    }
    return ret;
  }

  void FatropConicInterface::set_fatrop_conic_prob() {
    casadi_fatrop_setup(&p_);
  }

  int FatropConicInterface::init_mem(void* mem) const {
    if (Conic::init_mem(mem)) return 1;
    auto m = static_cast<FatropConicMemory*>(mem);

    m->add_stat("preprocessing");
    m->add_stat("solver");
    m->add_stat("postprocessing");
    return 0;
  }

  /** \brief Set the (persistent) work vectors */
  void FatropConicInterface::set_work(void* mem, const double**& arg, double**& res,
                          casadi_int*& iw, double*& w) const {

    auto m = static_cast<FatropConicMemory*>(mem);

    Conic::set_work(mem, arg, res, iw, w);

    m->d.prob = &p_;
    m->d.qp = &m->d_qp;
    casadi_fatrop_init(&m->d, &arg, &res, &iw, &w);
  }


  Dict FatropConicInterface::get_stats(void* mem) const {
    Dict stats = Conic::get_stats(mem);
    auto m = static_cast<FatropConicMemory*>(mem);

    stats["return_status"] = m->d.return_status;
    stats["iter_count"] = m->d.iter_count;
    return stats;
  }

  FatropConicMemory::FatropConicMemory() {
  }

  FatropConicMemory::~FatropConicMemory() {
  }

  Sparsity FatropConicInterface::blocksparsity(casadi_int rows, casadi_int cols,
      const std::vector<casadi_ocp_block>& blocks, bool eye) {
    DM r(rows, cols);
    for (auto && b : blocks) {
      if (eye) {
        r(range(b.offset_r, b.offset_r+b.rows),
          range(b.offset_c, b.offset_c+b.cols)) = DM::eye(b.rows);
        casadi_assert_dev(b.rows==b.cols);
      } else {
        r(range(b.offset_r, b.offset_r+b.rows),
        range(b.offset_c, b.offset_c+b.cols)) = DM::zeros(b.rows, b.cols);
      }
    }
    return r.sparsity();
  }
  void FatropConicInterface::blockptr(std::vector<double *>& vs, std::vector<double>& v,
      const std::vector<casadi_ocp_block>& blocks, bool eye) {
    casadi_int N = blocks.size();
    vs.resize(N);
    casadi_int offset=0;
    for (casadi_int k=0;k<N;++k) {
      vs[k] = get_ptr(v)+offset;
      if (eye) {
        casadi_assert_dev(blocks[k].rows==blocks[k].cols);
        offset+=blocks[k].rows;
      } else {
        offset+=blocks[k].rows*blocks[k].cols;
      }
    }
  }

  FatropConicInterface::FatropConicInterface(DeserializingStream& s) : Conic(s) {
    s.version("FatropConicInterface", 1);
  }

  void FatropConicInterface::serialize_body(SerializingStream &s) const {
    Conic::serialize_body(s);

    s.version("FatropConicInterface", 1);
  }


class CasadiStructuredQP : public fatrop::OCPAbstract {
  /// @brief number of states for time step k
  /// @param k: time step
  fatrop_int get_nxk(const fatrop_int k) const override {
    std::vector<fatrop_int> res = {2,3,2,1};
    return res[k];
  }
  /// @brief number of inputs for time step k
  /// @param k: time step
  fatrop_int get_nuk(const fatrop_int k) const override {
    std::vector<fatrop_int> res = {1, 2,1};
    return res[k];
  };
  /// @brief number of equality constraints for time step k
  /// @param k: time step
  fatrop_int get_ngk(const fatrop_int k) const override {
    std::vector<fatrop_int> res = {2, 0,0,0};
    return res[k]; // 2 from lbx

  };
  /// @brief  number of stage parameters for time step k
  /// @param k: time step
  fatrop_int get_n_stage_params_k(const fatrop_int k) const override { return 0;}
  /// @brief  number of global parameters
  fatrop_int get_n_global_params() const override { return 0;}
  /// @brief default stage parameters for time step k
  /// @param stage_params: pointer to array of size n_stage_params_k
  /// @param k: time step
  fatrop_int get_default_stage_paramsk(double *stage_params, const fatrop_int k) const override { return 0;}
  /// @brief default global parameters
  /// @param global_params: pointer to array of size n_global_params
  fatrop_int get_default_global_params(double *global_params) const override{ return 0; }
  /// @brief number of inequality constraints for time step k
  /// @param k: time step
  virtual fatrop_int get_ng_ineq_k(const fatrop_int k) const {
    std::vector<fatrop_int> res_lbg = {2, 1, 0, 0};
    std::vector<fatrop_int> res_lbx = {1, 5, 3, 1};
    return res_lbg[k]+res_lbx[k];
  }
  /// @brief horizon length
  fatrop_int get_horizon_length() const override { return 4; }
  /// @brief  discretized dynamics
  /// it evaluates the vertical concatenation of A_k^T, B_k^T, and b_k^T from the linearized dynamics x_{k+1} = A_k x_k + B_k u_k + b_k. 
  /// The matrix is in column major format.
  /// @param states_kp1: pointer to nx_{k+1}-array states of time step k+1
  /// @param inputs_k: pointer to array inputs of time step k
  /// @param states_k: pointer to array states of time step k
  /// @param stage_params_k: pointer to array stage parameters of time step k
  /// @param global_params: pointer to array global parameters
  /// @param res: pointer to (nu+nx+1 x nu+nx)-matrix 
  /// @param k: time step
  fatrop_int eval_BAbtk(
      const double *states_kp1,
      const double *inputs_k,
      const double *states_k,
      const double *stage_params_k,
      const double *global_params,
      MAT *res,
      const fatrop_int k) override {
        printf("eval_BAbtk k=%d\n", k);
        if (k==0) {
          std::vector<double> r = {1, 0.2, 1,   0,
                                      -0.1,	0.4,0 ,  0,
                                      0.3,	0.2,	0, 0};
          int out_m = 4; // rows
          int out_n = 3; // cols
          PACKMAT(out_m, out_n, get_ptr(r), out_m, res, 0, 0);
          blasfeo_print_dmat(out_m, out_n,  res, 0, 0);
        } else if (k==1) {
          std::vector<double> r = {1,	4,	2,	1,	0.3, 0,
                                    3,	1,	00,	1,	0.2, 0};
          int out_m = 6; // rows
          int out_n = 2; // cols
          PACKMAT(out_m, out_n, get_ptr(r), out_m, res, 0, 0);
          blasfeo_print_dmat(out_m, out_n,  res, 0, 0);
        } else if (k==2) {
          std::vector<double> r = {2, 4, 0, 0};
          int out_m = 4; // rows
          int out_n = 1; // cols
          PACKMAT(out_m, out_n, get_ptr(r), out_m, res, 0, 0);
          blasfeo_print_dmat(out_m, out_n,  res, 0, 0);
        }

        
      };
  /// @brief  stagewise Lagrangian Hessian
  /// It evaluates is the vertical concatenation of (1) the Hessian of the Lagrangian to the concatenation of (u_k, x_k) (2) the first order derivative of the Lagrangian Hessian to the concatenation of (u_k, x_k). 
  /// The matrix is in column major format.
  /// @param objective_scale: scale factor for objective function (usually 1.0)
  /// @param inputs_k: pointer to array inputs of time step k
  /// @param states_k: pointer to array states of time step k
  /// @param lam_dyn_k: pointer to array dual variables for dynamics of time step k
  /// @param lam_eq_k: pointer to array dual variables for equality constraints of time step k
  /// @param lam_eq_ineq_k: pointer to array dual variables for inequality constraints of time step k
  /// @param stage_params_k: pointer to array stage parameters of time step k
  /// @param global_params: pointer to array global parameters
  /// @param res: pointer to (nu+nx+1 x nu+nx)-matrix. 
  /// @param k
  /// @return
  fatrop_int eval_RSQrqtk(
      const double *objective_scale,
      const double *inputs_k,
      const double *states_k,
      const double *lam_dyn_k,
      const double *lam_eq_k,
      const double *lam_eq_ineq_k,
      const double *stage_params_k,
      const double *global_params,
      MAT *res,
      const fatrop_int k) override {

      }
  /// @brief stagewise equality constraints Jacobian. 
  /// It evaluates the vertical concatenation of (1) the Jacobian of the equality constraints to the concatenation of (u_k, x_k) (2) the equality constraints evaluated at u_k, x_k.
  /// The matrix is in column major format.
  /// @param inputs_k: pointer to array inputs of time step k
  /// @param states_k: pointer to array states of time step k
  /// @param stage_params_k: pointer to array stage parameters of time step k
  /// @param global_params: pointer to array global parameters
  /// @param res: pointer to (nu+nx+1 x ng)-matrix.
  /// @param k: time step
  /// @return
  fatrop_int eval_Ggtk(
      const double *inputs_k,
      const double *states_k,
      const double *stage_params_k,
      const double *global_params,
      MAT *res,
      const fatrop_int k) override {
    printf("eval_Ggtk k=%d\n", k);
    if (k==0) {
      std::vector<double> r = {2, 0, 0.3, 0,
                              1, 1, 0.4, 0};
      int out_m = 4; // rows
      int out_n = 2; // cols
      PACKMAT(out_m, out_n, get_ptr(r), out_m, res, 0, 0);
      blasfeo_print_dmat(out_m, out_n,  res, 0, 0);
    }
  }
  /// @brief stagewise inequality constraints Jacobian. 
  /// It evaluates the vertical concatenation of (1) the Jacobian of the inequality constraints to the concatenation of (u_k, x_k) (2) the inequality constraints evaluated at u_k, x_k. 
  /// The matrix is in column major format.
  /// @param inputs_k: pointer to array inputs of time step k
  /// @param states_k: pointer to array states of time step k
  /// @param stage_params_k: pointer to array stage parameters of time step k
  /// @param global_params_ko: pointer to array global parameters
  /// @param res: pointer to (nu+nx+1 x ng_ineq)-matrix, column major format
  /// @param k : time step
  /// @return
  fatrop_int eval_Ggt_ineqk(
      const double *inputs_k,
      const double *states_k,
      const double *stage_params_k,
      const double *global_params,
      MAT *res,
      const fatrop_int k) {
    printf("eval_Ggt_ineqk k=%d\n", k);
  }
  /// @brief the dynamics constraint violation (b_k = -x_{k+1} + f_k(u_k, x_k, p_k, p))
  /// @param states_kp1: pointer to array states of time step k+1
  /// @param inputs_k: pointer to array inputs of time step k
  /// @param states_k: pointer to array states of time step k
  /// @param stage_params_k: pointer to array stage parameters of time step k
  /// @param global_params: pointer to array global parameters
  /// @param res: pointer to array nx_{k+1}-vector
  /// @param k: time step
  /// @return
  fatrop_int eval_bk(
      const double *states_kp1,
      const double *inputs_k,
      const double *states_k,
      const double *stage_params_k,
      const double *global_params,
      double *res,
      const fatrop_int k) override {
  printf("eval_bk k=%d\n", k);
      }
  /// @brief the equality constraint violation (g_k = g_k(u_k, x_k, p_k, p))
  /// @param inputs_k: pointer to array inputs of time step k
  /// @param states_k: pointer to array states of time step k
  /// @param stage_params_k: pointer to array stage parameters of time step k
  /// @param global_params: pointer to array global parameters
  /// @param res: pointer to array ng-vector
  /// @param k: time step
  fatrop_int eval_gk(
      const double *states_k,
      const double *inputs_k,
      const double *stage_params_k,
      const double *global_params,
      double *res,
      const fatrop_int k) override {
    printf("eval_gk k=%d\n", k);
  }
  /// @brief the inequality constraint violation (g_ineq_k = g_ineq_k(u_k, x_k, p_k, p))
  /// @param inputs_k: pointer to array inputs of time step k
  /// @param states_k: pointer to array states of time step k
  /// @param stage_params_k: pointer to array stage parameters of time step k
  /// @param global_params: pointer to array global parameters
  /// @param res: pointer to array ng_ineq-vector
  /// @param k: time step
  fatrop_int eval_gineqk(
      const double *states_k,
      const double *inputs_k,
      const double *stage_params_k,
      const double *global_params,
      double *res,
      const fatrop_int k)  override {
      printf("eval_gineqk k=%d\n", k);
  }
  /// @brief gradient of the objective function (not the Lagrangian!) to the concatenation of (u_k, x_k)
  /// @param objective_scale: pointer to objective scale
  /// @param inputs_k: pointer to array inputs of time step k
  /// @param states_k: pointer to array states of time step k
  /// @param stage_params_k: pointer to array stage parameters of time step k
  /// @param global_params: pointer to array global parameters
  /// @param res: pointer to (nu+nx)-array
  /// @param k: time step
  fatrop_int eval_rqk(
      const double *objective_scale,
      const double *inputs_k,
      const double *states_k,
      const double *stage_params_k,
      const double *global_params,
      double *res,
      const fatrop_int k) override {
      printf("eval_rqk k=%d\n", k);
  }
  /// @brief objective function value 
  /// @param objective_scale: pointer to array objective scale
  /// @param inputs_k: pointer to array inputs of time step k
  /// @param states_k: pointer to array states of time step k
  /// @param stage_params_k: pointer to array stage parameters of time step k
  /// @param global_params: pointer to array global parameters
  /// @param res: pointer to double
  /// @param k: time step
  fatrop_int eval_Lk(
      const double *objective_scale,
      const double *inputs_k,
      const double *states_k,
      const double *stage_params_k,
      const double *global_params,
      double *res,
      const fatrop_int k) override {
      printf("eval_Lk k=%d\n", k);
  }
  /// @brief the bounds of the inequalites at stage k
  /// @param lower: pointer to ng_ineq-vector
  /// @param upper: pointer to ng_ineq-vector
  /// @param k: time step
  fatrop_int get_boundsk(double *lower, double *upper, const fatrop_int k) const override {
      printf("get_boundsk k=%d\n", k);
  }
  /// @brief default initial guess for the states of stage k
  /// @param xk: pointer to states of time step k 
  /// @param k: time step
  fatrop_int get_initial_xk(double *xk, const fatrop_int k) const override {
    printf("get_initial_xk k=%d\n", k);
  }
  /// @brief default initial guess for the inputs of stage k
  /// @param uk: pointer to inputs of time step k
  /// @param k: time step
  fatrop_int get_initial_uk(double *uk, const fatrop_int k) const override {
    printf("get_initial_uk k=%d\n", k);
  }

};


  int FatropConicInterface::
  solve(const double** arg, double** res, casadi_int* iw, double* w, void* mem) const {
    auto m = static_cast<FatropConicMemory*>(mem);

    // Statistics
    m->fstats.at("solver").tic();

    casadi::CasadiStructuredQP qp;

    fatrop::OCPApplication app(std::make_shared<casadi::CasadiStructuredQP>(qp));
    app.build();

    app.optimize();

    m->fstats.at("solver").toc();


    return 0;
  }

} // namespace casadi
