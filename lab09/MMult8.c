/* OpenMP implementation of multi-threaded block matrix multiplication */
/* Split the matrix into fixed size blocks */
#include "dgemm.h"
#include <immintrin.h>

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

#define BLOCK_SIZE 256

inline static void naive_dgemm(int m, int n, int k, double *a, int lda, 
                               double *b, int ldb, double *c, int ldc);

/* Routine for computing C = A * B + C */
/* (m*n) = (m*k) * (k*n) */
void MY_MMult(int m, int n, int k, double *a, int lda, 
              double *b, int ldb, double *c, int ldc) {
    for (int i = 0; i < m; i += BLOCK_SIZE) {
        int len_i = BLOCK_SIZE < m - i ? BLOCK_SIZE : m - i;
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            int len_j = BLOCK_SIZE < n - j ? BLOCK_SIZE : n - j;
            for (int p = 0; p < k; p += BLOCK_SIZE) {
                int len_p = BLOCK_SIZE < k - p ? BLOCK_SIZE : k - p;
                naive_dgemm(len_i, len_j, len_p, &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
            }
        }
    }
}

inline static void naive_dgemm(int m, int n, int k, double *a, int lda, 
                               double *b, int ldb, double *c, int ldc) {
    int j;
    for (j = 0; j <= n - 4; j += 4) {
        int i;
        for (i = 0; i <= m - 2; i += 2) {
            __m128d c_i01j0, c_i01j1, c_i01j2, c_i01j3;
            c_i01j0 = _mm_loadu_pd(&C(i, j));
            c_i01j1 = _mm_loadu_pd(&C(i, j + 1));
            c_i01j2 = _mm_loadu_pd(&C(i, j + 2));
            c_i01j3 = _mm_loadu_pd(&C(i, j + 3));

            for (int p = 0; p < k; p++) {
                __m128d b_p0j0, b_p0j1, b_p0j2, b_p0j3;
                b_p0j0 = _mm_loaddup_pd(&B(p, j));
                b_p0j1 = _mm_loaddup_pd(&B(p, j + 1));
                b_p0j2 = _mm_loaddup_pd(&B(p, j + 2));
                b_p0j3 = _mm_loaddup_pd(&B(p, j + 3));
                __m128d a_i01p0;
                a_i01p0 = _mm_loadu_pd(&A(i, p));

                c_i01j0 = _mm_fmadd_pd(a_i01p0, b_p0j0, c_i01j0);
                c_i01j1 = _mm_fmadd_pd(a_i01p0, b_p0j1, c_i01j1);
                c_i01j2 = _mm_fmadd_pd(a_i01p0, b_p0j2, c_i01j2);
                c_i01j3 = _mm_fmadd_pd(a_i01p0, b_p0j3, c_i01j3);
            }
            _mm_storeu_pd(&C(i, j),     c_i01j0);
            _mm_storeu_pd(&C(i, j + 1), c_i01j1);
            _mm_storeu_pd(&C(i, j + 2), c_i01j2);
            _mm_storeu_pd(&C(i, j + 3), c_i01j3);
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
