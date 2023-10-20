/* Block + AVX2/FMA 32 elements a time (modified block size) */
#include "dgemm.h"
#include <immintrin.h>

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

#define Mc 64
#define Nc 256
#define Kc 64
#define Mr 8
#define Nr 4

inline static void inner_dgemm(int m, int n, int k, double *a, int lda, 
                               double *b, int ldb, double *c, int ldc);

inline static void unit_dgemm(int k, double *a, int lda,
                              double *b, int ldb, double *c, int ldc);

/* Routine for computing C = A * B + C */
/* (m*n) = (m*k) * (k*n) */
void MY_MMult(int m, int n, int k, double *a, int lda, 
              double *b, int ldb, double *c, int ldc) {
    int lda_aligned = (m + 7) & 0xfffffff8;
    double *a_aligned = (double *) aligned_alloc(64, lda_aligned * k * sizeof(double));
    #define A_ALN(i, j) a_aligned[(i) + lda_aligned * (j)]
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < m; i++) {
            A_ALN(i, j) = A(i, j);
        }
    }
    int ldc_aligned = (m + 7) & 0xfffffff8;
    double *c_aligned = (double *) aligned_alloc(64, ldc_aligned * n * sizeof(double));
    #define C_ALN(i, j) c_aligned[(i) + ldc_aligned * (j)]
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            C_ALN(i, j) = C(i, j);
        }
    }

    #pragma omp parallel for num_threads(16)
    for (int j = 0; j < n; j += Nc) {
        int len_j = Nc < n - j ? Nc : n - j;
        for (int p = 0; p < k; p += Kc) {
            int len_p = Kc < k - p ? Kc : k - p;
            for (int i = 0; i < m; i += Mc) {
                int len_i = Mc < m - i ? Mc : m - i;
                inner_dgemm(len_i, len_j, len_p, &A_ALN(i, p), lda_aligned,
                            &B(p, j), ldb, &C_ALN(i, j), ldc_aligned);
            }
        }
    }

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            C(i, j) = C_ALN(i, j);
        }
    }
    free(a_aligned); free(c_aligned);
}

inline static void inner_dgemm(int m, int n, int k, double *a, int lda, 
                               double *b, int ldb, double *c, int ldc) {
    int j;
    for (j = 0; j <= n - Nr; j += Nr) {
        int i;
        for (i = 0; i <= m - Mr; i += Mr) {
            unit_dgemm(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
        for (; i < m; i++) {
            register
            double c_i0j0 = C(i, j),     c_i0j1 = C(i, j + 1),
                   c_i0j2 = C(i, j + 2), c_i0j3 = C(i, j + 3);
            for (int p = 0; p < k; p++) {
                c_i0j0 += A(i, p) * B(p, j);
                c_i0j1 += A(i, p) * B(p, j + 1);
                c_i0j2 += A(i, p) * B(p, j + 2);
                c_i0j3 += A(i, p) * B(p, j + 3);
            }
            C(i, j)     = c_i0j0; C(i, j + 1) = c_i0j1;
            C(i, j + 2) = c_i0j2; C(i, j + 3) = c_i0j3;
        }
    }
    for (; j < n; j++) {
        for (int i = 0; i < m; i++) {
            register double c_ij = C(i, j);
            for (int p = 0; p < k; p++) {
                c_ij += A(i, p) * B(p, j);
            }
            C(i, j) = c_ij;
        }
    }
}

inline static void unit_dgemm(int k, double *a, int lda,
                              double *b, int ldb, double *c, int ldc) {
    __m256d c_i0123j0, c_i0123j1, c_i0123j2, c_i0123j3,
            c_i4567j0, c_i4567j1, c_i4567j2, c_i4567j3;
    c_i0123j0 = _mm256_load_pd(&C(0, 0));
    c_i0123j1 = _mm256_load_pd(&C(0, 1));
    c_i0123j2 = _mm256_load_pd(&C(0, 2));
    c_i0123j3 = _mm256_load_pd(&C(0, 3));
    c_i4567j0 = _mm256_load_pd(&C(4, 0));
    c_i4567j1 = _mm256_load_pd(&C(4, 1));
    c_i4567j2 = _mm256_load_pd(&C(4, 2));
    c_i4567j3 = _mm256_load_pd(&C(4, 3));

    for (int p = 0; p < k; p++) {
        __m256d b_p0j0, b_p0j1, b_p0j2, b_p0j3;
        b_p0j0 = _mm256_set1_pd(B(p, 0));
        b_p0j1 = _mm256_set1_pd(B(p, 1));
        b_p0j2 = _mm256_set1_pd(B(p, 2));
        b_p0j3 = _mm256_set1_pd(B(p, 3));
        __m256d a_i0123p0, a_i4567p0;
        a_i0123p0 = _mm256_load_pd(&A(0, p));
        a_i4567p0 = _mm256_load_pd(&A(4, p));

        c_i0123j0 = _mm256_fmadd_pd(a_i0123p0, b_p0j0, c_i0123j0);
        c_i0123j1 = _mm256_fmadd_pd(a_i0123p0, b_p0j1, c_i0123j1);
        c_i0123j2 = _mm256_fmadd_pd(a_i0123p0, b_p0j2, c_i0123j2);
        c_i0123j3 = _mm256_fmadd_pd(a_i0123p0, b_p0j3, c_i0123j3);
        c_i4567j0 = _mm256_fmadd_pd(a_i4567p0, b_p0j0, c_i4567j0);
        c_i4567j1 = _mm256_fmadd_pd(a_i4567p0, b_p0j1, c_i4567j1);
        c_i4567j2 = _mm256_fmadd_pd(a_i4567p0, b_p0j2, c_i4567j2);
        c_i4567j3 = _mm256_fmadd_pd(a_i4567p0, b_p0j3, c_i4567j3);
    }
    _mm256_store_pd(&C(0, 0), c_i0123j0);
    _mm256_store_pd(&C(0, 1), c_i0123j1);
    _mm256_store_pd(&C(0, 2), c_i0123j2);
    _mm256_store_pd(&C(0, 3), c_i0123j3);
    _mm256_store_pd(&C(4, 0), c_i4567j0);
    _mm256_store_pd(&C(4, 1), c_i4567j1);
    _mm256_store_pd(&C(4, 2), c_i4567j2);
    _mm256_store_pd(&C(4, 3), c_i4567j3);
}
