//
//    MIT No Attribution
//
//    Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl, KU Leuven.
//
//    Permission is hereby granted, free of charge, to any person obtaining a copy of this
//    software and associated documentation files (the "Software"), to deal in the Software
//    without restriction, including without limitation the rights to use, copy, modify,
//    merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
//    permit persons to whom the Software is furnished to do so.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
//    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
//    PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
//    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
//    OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

// C-REPLACE "casadi_qp_prob<T1>" "struct casadi_qp_prob"
// C-REPLACE "casadi_qp_data<T1>" "struct casadi_qp_data"

// C-REPLACE "reinterpret_cast<int**>" "(int**) "
// C-REPLACE "reinterpret_cast<int*>" "(int*) "
// C-REPLACE "const_cast<int*>" "(int*) "


// SYMBOL "fatrop_mproject"
template<typename T1>
void casadi_fatrop_mproject(T1 factor, const T1* x, const casadi_int* sp_x,
                            T1* y, const casadi_int* sp_y, T1* w) {
    casadi_int ncol_y;
    const casadi_int* colind_y;
    ncol_y = sp_y[1];
    colind_y = sp_y+2;
    casadi_project(x, sp_x, y, sp_y, w);
    casadi_scal(colind_y[ncol_y], factor, y);
}

// SYMBOL "fatrop_dense_transfer"
template<typename T1>
void casadi_fatrop_dense_transfer(double factor, const T1* x,
                                    const casadi_int* sp_x, T1* y,
                                    const casadi_int* sp_y, T1* w) {
    casadi_sparsify(x, w, sp_x, 0);
    casadi_int nrow_y = sp_y[0];
    casadi_int ncol_y = sp_y[1];
    const casadi_int *colind_y = sp_y+2, *row_y = sp_y + 2 + ncol_y+1;
    /* Loop over columns of y */
    casadi_int i, el;
    for (i=0; i<ncol_y; ++i) {
        for (el=colind_y[i]; el<colind_y[i+1]; ++el) y[nrow_y*i + row_y[el]] += factor*(*w++);
    }
}

// SYMBOL "fatrop_block"
struct casadi_ocp_block {
    casadi_int offset_r;
    casadi_int offset_c;
    casadi_int rows;
    casadi_int cols;
};
// C-REPLACE "casadi_ocp_block" "struct casadi_ocp_block"

// SYMBOL "fatrop_unpack_blocks"
inline void casadi_fatrop_unpack_blocks(casadi_int N, casadi_ocp_block* blocks, const casadi_int* packed) {
    casadi_int i;
    for (i=0;i<N;++i) {
        blocks[i].offset_r = *packed++;
        blocks[i].offset_c = *packed++;
        blocks[i].rows = *packed++;
        blocks[i].cols = *packed++;
    }
}

// SYMBOL "fatrop_ptr_block"
template<typename T1>
void casadi_fatrop_ptr_block(casadi_int N, T1** vs, T1* v, const casadi_ocp_block* blocks, int eye) {
    casadi_int k, offset = 0;
    for(k=0;k<N;++k) {
        vs[k] = v+offset;
        if (eye) {
        offset += blocks[k].rows;
        } else {
        offset += blocks[k].rows*blocks[k].cols;
        }
    }
}

template<typename T1>
struct casadi_fatrop_conic_prob {
  const casadi_qp_prob<T1>* qp;
  const int *nx, *nu, *ng;
  const int *nbx, *nbu, *ns;
  const int *nsbx, *nsbu, *nsg;
  // Sparsities
  const casadi_int *sp_x, *sp_ba;
  const casadi_int *Asp, *Bsp, *Csp, *Dsp;
  const casadi_int *Rsp, *Isp, *Ssp, *Qsp;
  const casadi_int *bsp;
  const casadi_int *xsp, *usp;
  const casadi_int *pisp;
  const casadi_int *theirs_xsp, *theirs_usp, *theirs_Xsp, *theirs_Usp;
  const casadi_int *lamg_gapsp, *lugsp;

  casadi_int N;
  casadi_int nx_total, nu_total, ng_total;
  casadi_ocp_block *A, *B, *C, *D;
  casadi_ocp_block *R, *I, *S, *Q;
  casadi_ocp_block *b, *lug;
  casadi_ocp_block *u, *x;
  casadi_ocp_block *lam_ul, *lam_xl, *lam_uu, *lam_xu, *lam_cl, *lam_cu;

  T1 warm_start;
  T1 inf;

};
// C-REPLACE "casadi_fatrop_conic_prob<T1>" "struct casadi_fatrop_conic_prob"

// SYMBOL "fatrop_setup"
template<typename T1>
void casadi_fatrop_setup(casadi_fatrop_conic_prob<T1>* p) {


}



// SYMBOL "fatrop_data"
template<typename T1>
struct casadi_fatrop_conic_data {
  // Problem structure
  const casadi_fatrop_conic_prob<T1>* prob;
  // Problem structure
  casadi_qp_data<T1>* qp;
  T1 *A, *B, *C, *D;
  T1 *R, *I, *Q, *S;
  T1 *b, *b2;
  T1 *x, *q;
  T1 *u, *r;
  T1 *lg, *ug;
  T1 *pi;
  T1 *lbx, *ubx, *lbu, *ubu, *lam;

  T1 **hA, **hB, **hC, **hD;
  T1 **hR, **hI, **hQ, **hS;
  T1 **hx, **hq;
  T1 **hu, **hr;
  T1 **hlg, **hug;
  T1 **hb;

  T1 **hZl, **hZu, **hzl, **hzu, **hlls, **hlus;
  T1 **pis, **hlbx, **hubx, **hlbu, **hubu, **lams;

  int *iidxbx, *iidxbu;
  int **hidxbx, **hidxbu, **hidxs;

  int iter_count;
  int return_status;
  T1 res_stat;
  T1 res_eq;
  T1 res_ineq;
  T1 res_comp;

  T1 *pv;
};
// C-REPLACE "casadi_fatrop_conic_data<T1>" "struct casadi_fatrop_conic_data"


// SYMBOL "qp_work"
template<typename T1>
void casadi_fatrop_work(const casadi_fatrop_conic_prob<T1>* p, casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w) {
  casadi_qp_work(p->qp, sz_arg, sz_res, sz_iw, sz_w);

}

// SYMBOL "qp_init"
template<typename T1>
void casadi_fatrop_init(casadi_fatrop_conic_data<T1>* d, const T1*** arg, T1*** res, casadi_int** iw, T1** w) {
  // Local variables
  casadi_int offset, i, k;
  
  const casadi_fatrop_conic_prob<T1>* p = d->prob;

}