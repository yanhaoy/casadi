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

    /* Disassemble A input into:
       A B I
       C D
           A B I
           C D
    */
    casadi_int offset_r = 0, offset_c = 0;
    for (casadi_int k=0;k<N_;++k) { // Loop over blocks
      AB_blocks.push_back({offset_r,        offset_c,            nx[k+1], nx[k]+nu[k]});
      CD_blocks.push_back({offset_r+nx[k+1], offset_c,           ng[k], nx[k]+nu[k]});
      offset_c+= nx[k]+nu[k];
      if (k+1<N_)
        I_blocks.push_back({offset_r, offset_c, nx[k+1], nx[k+1]});
      else
        I_blocks.push_back({offset_r, offset_c, nx[k+1], nx[k+1]});
      offset_r+= nx[k+1]+ng[k];
    }

    casadi_int offset = 0;
    AB_offsets_.push_back(0);
    for (auto e : AB_blocks) {
      offset += e.rows*e.cols;
      AB_offsets_.push_back(offset);
    }
    offset = 0;
    CD_offsets_.push_back(0);
    for (auto e : CD_blocks) {
      offset += e.rows*e.cols;
      CD_offsets_.push_back(offset);
    }

    ABsp_ = blocksparsity(na_, nx_, AB_blocks);
    CDsp_ = blocksparsity(na_, nx_, CD_blocks);

    /* Disassemble H input into:
       Q S'
       S R
           Q S'
           S R

       Multiply by 2
    */
    offset = 0;
    for (casadi_int k=0;k<N_+1;++k) { // Loop over blocks
      RSQ_blocks.push_back({offset, offset,       nx[k]+nu[k], nx[k]+nu[k]});
      offset+= nx[k]+nu[k];
    }
    RSQsp_ = blocksparsity(nx_, nx_, RSQ_blocks);

    offset = 0;
    RSQ_offsets_.push_back(0);
    for (auto e : RSQ_blocks) {
      offset += e.rows*e.cols;
      RSQ_offsets_.push_back(offset);
    }
    set_fatrop_conic_prob();

    // Allocate memory
    casadi_int sz_arg, sz_res, sz_w, sz_iw;
    casadi_fatrop_conic_work(&p_, &sz_arg, &sz_res, &sz_iw, &sz_w);

    uout() << sz_arg << std::endl;
    uout() << sz_res << std::endl;
    uout() << sz_iw << std::endl;
    uout() << sz_w << std::endl;

    alloc_arg(sz_arg, true);
    alloc_res(sz_res, true);
    alloc_iw(sz_iw, true);
    alloc_w(sz_w, true);



  }

  void FatropConicInterface::set_temp(void* mem, const double** arg, double** res,
                      casadi_int* iw, double* w) const {

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
    p_.qp = &p_qp_;
    p_.nx  = get_ptr(nxs_);
    p_.nu  = get_ptr(nus_);
    p_.ABsp = ABsp_;
    p_.AB_offsets = get_ptr(AB_offsets_);
    p_.CDsp = CDsp_;
    p_.CD_offsets = get_ptr(CD_offsets_);
    p_.RSQsp = RSQsp_;
    p_.RSQ_offsets = get_ptr(RSQ_offsets_);
    p_.AB = get_ptr(AB_blocks);
    p_.CD = get_ptr(CD_blocks);
    p_.RSQ = get_ptr(RSQ_blocks);
    p_.N = N_;
    casadi_fatrop_conic_setup(&p_);
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

    casadi_qp_data<double>* d_qp = m->d.qp;

    uout() << p_.qp->na << std::endl;
    uout() << p_.qp->nx << std::endl;

    uout() << std::vector<double>(d_qp->lba,d_qp->lba+p_.qp->na) << std::endl;
    uout() << std::vector<double>(d_qp->uba,d_qp->uba+p_.qp->na) << std::endl;
    uout() << std::vector<double>(d_qp->lbx,d_qp->lbx+p_.qp->nx) << std::endl;
    uout() << std::vector<double>(d_qp->ubx,d_qp->ubx+p_.qp->nx) << std::endl;

    casadi_fatrop_conic_set_work(&m->d, &arg, &res, &iw, &w);
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
  public:
    const FatropConicInterface& solver;
    FatropConicMemory* m;
    CasadiStructuredQP(const FatropConicInterface& solv, FatropConicMemory* mem) : solver(solv), m(mem) {

    }
  /// @brief number of states for time step k
  /// @param k: time step
  fatrop_int get_nxk(const fatrop_int k) const override {
    if (k==solver.nxs_.size()) return 0;
    printf("get_nxk k=%d %d\n", k, solver.nxs_[k]);
    return solver.nxs_[k];
  }
  /// @brief number of inputs for time step k
  /// @param k: time step
  fatrop_int get_nuk(const fatrop_int k) const override {
    printf("get_nuk k=%d %d\n", k, solver.nus_[k]);
    return solver.nus_[k];
  };
  /// @brief number of equality constraints for time step k
  /// @param k: time step
  fatrop_int get_ngk(const fatrop_int k) const override {
    auto d = &m->d;
    auto p = d->prob;
    int ret;
    if (k==p->N) {
      ret = 0;
    } else {
      fatrop_int n_a_eq = d->a_eq_idx[k+1]-d->a_eq_idx[k];
      fatrop_int n_x_eq = d->x_eq_idx[k+1]-d->x_eq_idx[k];
      uout() << "debug" << n_a_eq << ":" << n_x_eq << std::endl;

      ret = n_a_eq+n_x_eq;
    }
    printf("get_ngk k=%d: %d\n", k, ret);
    return ret;
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
    int ret;
    auto d = &m->d;
    auto p = d->prob;
    if (k==p->N) {
      ret = 0;
    } else {
      fatrop_int n_a_ineq = d->a_ineq_idx[k+1]-d->a_ineq_idx[k];
      fatrop_int n_x_ineq = d->x_ineq_idx[k+1]-d->x_ineq_idx[k];
      ret = n_a_ineq+n_x_ineq;
    }
    printf("get_ng_ineqk k=%d: %d\n", k, ret);
    return ret;
  }
  /// @brief horizon length
  fatrop_int get_horizon_length() const override { return solver.nxs_.size(); }
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
        auto d = &m->d;
        auto p = d->prob;
        casadi_qp_data<double>* d_qp = d->qp;
        printf("eval_BAbtk k=%d\n", k);
        blasfeo_pack_tran_dmat(p->nx[k+1], p->nx[k]+p->nu[k], d->AB+p->AB_offsets[k], p->nx[k+1], res, 0, 0);
        uout() << d_qp->lba << std::endl;
        uout() << p->AB[k].offset_r << std::endl;
        blasfeo_pack_dmat(1, p->nx[k+1], const_cast<double*>(d_qp->lba+p->AB[k].offset_r), 1, res, p->nx[k+1], 0);
        blasfeo_print_dmat(p->nx[k]+p->nu[k]+1, p->nx[k+1], res, 0, 0);
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
        printf("eval_RSQrqtk k=%d\n", k);
        auto d = &m->d;
        auto p = d->prob;
        casadi_qp_data<double>* d_qp = d->qp;

        int n = p->nx[k]+p->nu[k];
        blasfeo_pack_dmat(n, n, d->RSQ+p->RSQ_offsets[k], n, res, 0, 0);

        blasfeo_print_dmat(n+1, n, res, 0, 0);
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
    casadi_int i, column;
    double one = 1;
    printf("eval_Ggt_k k=%d\n", k);
    auto d = &m->d;
    auto p = d->prob;
    casadi_qp_data<double>* d_qp = d->qp;

    int n_a_eq = d->a_eq_idx[k+1]-d->a_eq_idx[k];
    int n_x_eq = d->x_eq_idx[k+1]-d->x_eq_idx[k];
    int ng_eq = n_a_eq+n_x_eq;

    column = 0;
    int n = p->nx[k]+p->nu[k];
    for (i=d->a_eq_idx[k];i<d->a_eq_idx[k+1];++i) {
      blasfeo_pack_tran_dmat(1, n, d->CD+p->CD_offsets[k]+(d->a_eq[i]-p->CD[k].offset_r), n_a_eq, res, 0, column++);
    }
    for (i=d->x_eq_idx[k];i<d->x_eq_idx[k+1];++i) {
      blasfeo_pack_tran_dmat(1, 1, &one, 1, res, d->x_eq[i]-p->CD[k].offset_c, column++);
    }
    blasfeo_print_dmat(p->nx[k]+p->nu[k]+1, ng_eq,  res, 0, 0);
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
    casadi_int i, column;
    double one = 1;
    printf("eval_Ggt_ineqk k=%d\n", k);
    auto d = &m->d;
    auto p = d->prob;
    casadi_qp_data<double>* d_qp = d->qp;

    int n_a_ineq = d->a_ineq_idx[k+1]-d->a_ineq_idx[k];
    int n_x_ineq = d->x_ineq_idx[k+1]-d->x_ineq_idx[k];
    int ng_ineq = n_a_ineq+n_x_ineq;

    column = 0;
    int n = p->nx[k]+p->nu[k];
    for (i=d->a_ineq_idx[k];i<d->a_ineq_idx[k+1];++i) {
      blasfeo_pack_tran_dmat(1, n, d->CD+p->CD_offsets[k]+(d->a_ineq[i]-p->CD[k].offset_r), n_a_ineq, res, 0, column++);
    }
    for (i=d->x_ineq_idx[k];i<d->x_ineq_idx[k+1];++i) {
      blasfeo_pack_tran_dmat(1, 1, &one, 1, res, d->x_ineq[i]-p->CD[k].offset_c, column++);
    }
    blasfeo_print_dmat(p->nx[k]+p->nu[k]+1, ng_ineq,  res, 0, 0);
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
      auto d = &m->d;
      auto p = d->prob;
      uout() << std::vector<double>(res,res+p->nx[k+1]) << std::endl;
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

      uout() << std::vector<double>(res,res+get_ng_ineq_k(k)) << std::endl;
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
      auto d = &m->d;
      auto p = d->prob;
      uout() << std::vector<double>(res,res+p->nx[k]+p->nu[k]) << std::endl;
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
      auto d = &m->d;
      auto p = d->prob;
      casadi_qp_data<double>* d_qp = d->qp;
      if (k==p->N) return 0;
      int i=0;
      int start = d->a_ineq_idx[k];
      for (i=start;i<d->a_ineq_idx[k+1];++i) {
        lower[i-start] = d_qp->lba[d->a_ineq[i]];
        upper[i-start] = d_qp->uba[d->a_ineq[i]];
      }
      start = d->x_ineq_idx[k];
      int offset = d->a_ineq_idx[k+1]-d->a_ineq_idx[k];
      for (i=start;i<d->x_ineq_idx[k+1];++i) {
        lower[i-start+offset] = d_qp->lbx[d->x_ineq[i]];
        upper[i-start+offset] = d_qp->ubx[d->x_ineq[i]];
      }

      fatrop_int n_a_ineq = d->a_ineq_idx[k+1]-d->a_ineq_idx[k];
      fatrop_int n_x_ineq = d->x_ineq_idx[k+1]-d->x_ineq_idx[k];
      fatrop_int ng_ineq = n_a_ineq+n_x_ineq;

      uout() << "lower" << std::vector<double>(lower,lower+ng_ineq) << std::endl;
      uout() << "upper" << std::vector<double>(upper,upper+ng_ineq) << std::endl;
  }
  /// @brief default initial guess for the states of stage k
  /// @param xk: pointer to states of time step k 
  /// @param k: time step
  fatrop_int get_initial_xk(double *xk, const fatrop_int k) const override {
    printf("get_initial_xk k=%d\n", k);
    auto d = &m->d;
    auto p = d->prob;
    casadi_qp_data<double>* d_qp = d->qp;
    if (k==p->N) return 0;
    casadi_copy(d_qp->x0+p->CD[k].offset_c, p->nx[k], xk);
  }
  /// @brief default initial guess for the inputs of stage k
  /// @param uk: pointer to inputs of time step k
  /// @param k: time step
  fatrop_int get_initial_uk(double *uk, const fatrop_int k) const override {
    printf("get_initial_uk k=%d\n", k);
    auto d = &m->d;
    auto p = d->prob;
    casadi_qp_data<double>* d_qp = d->qp;
    if (k==p->N) return 0;
    casadi_copy(d_qp->x0+p->CD[k].offset_c+p->nx[k], p->nu[k], uk);
  }

};


  int FatropConicInterface::
  solve(const double** arg, double** res, casadi_int* iw, double* w, void* mem) const {
    auto m = static_cast<FatropConicMemory*>(mem);


    casadi_fatrop_conic_solve(&m->d, arg, res, iw, w);

    // Statistics
    m->fstats.at("solver").tic();

    casadi::CasadiStructuredQP qp(*this, m);

    fatrop::OCPApplication app(std::make_shared<casadi::CasadiStructuredQP>(qp));
    app.build();

    app.optimize();

    m->fstats.at("solver").toc();


    return 0;
  }

} // namespace casadi
