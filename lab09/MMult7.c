/* Block + jip 16 elements a time */
#include "dgemm.h"

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

#define BLOCK_SIZE 64

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
            register
            double c_i0j0 = C(i, j),         c_i0j1 = C(i, j + 1),
                   c_i0j2 = C(i, j + 2),     c_i0j3 = C(i, j + 3),
                   c_i1j0 = C(i + 1, j),     c_i1j1 = C(i + 1, j + 1),
                   c_i1j2 = C(i + 1, j + 2), c_i1j3 = C(i + 1, j + 3);
            for (int p = 0; p < k; p++) {
                register
                double a_i0p0 = A(i, p), a_i1p0 = A(i + 1, p);
                c_i0j0 += a_i0p0 * B(p, j);
                c_i0j1 += a_i0p0 * B(p, j + 1);
                c_i0j2 += a_i0p0 * B(p, j + 2);
                c_i0j3 += a_i0p0 * B(p, j + 3);
                c_i1j0 += a_i1p0 * B(p, j);
                c_i1j1 += a_i1p0 * B(p, j + 1);
                c_i1j2 += a_i1p0 * B(p, j + 2);
                c_i1j3 += a_i1p0 * B(p, j + 3);
            }
            C(i, j)         = c_i0j0, C(i, j + 1)     = c_i0j1,
            C(i, j + 2)     = c_i0j2, C(i, j + 3)     = c_i0j3,
            C(i + 1, j)     = c_i1j0, C(i + 1, j + 1) = c_i1j1,
            C(i + 1, j + 2) = c_i1j2, C(i + 1, j + 3) = c_i1j3;
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
