/* OpenMP implementation of multi-threaded block matrix multiplication */
/* Split the matrix into fixed size blocks */
#include "dgemm.h"

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

#define BLOCK_SIZE 256

inline static void block_dgemm(int m, int n, int k, double *a, int lda, 
                                      double *b, int ldb,
                                      double *c, int ldc);

inline static void naive_dgemm(int m, int n, int k, double *a, int lda, 
                                      double *b, int ldb,
                                      double *c, int ldc);

/* Routine for computing C = A * B + C */
/* (m*n) = (m*k) * (k*n) */
void MY_MMult(int m, int n, int k, double *a, int lda, 
                                   double *b, int ldb,
                                   double *c, int ldc) {
    block_dgemm(m, n, k, a, lda, b, ldb, c, ldc);
}

inline static void block_dgemm(int m, int n, int k, double *a, int lda, 
                        double *b, int ldb,
                        double *c, int ldc) {
    int i = 0, j = 0, p = 0;
    for (i = 0; i <= m - BLOCK_SIZE; i += BLOCK_SIZE) {
        for (j = 0; j <= n - BLOCK_SIZE; j += BLOCK_SIZE) {
            for (p = 0; p <= k - BLOCK_SIZE; p += BLOCK_SIZE) {
                naive_dgemm(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE,
                            &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
            }
            naive_dgemm(BLOCK_SIZE, BLOCK_SIZE, k - p,
                        &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
        }
    }
    for (j = 0; j <= n - BLOCK_SIZE; j += BLOCK_SIZE) {
        for (p = 0; p <= k - BLOCK_SIZE; p += BLOCK_SIZE) {
            naive_dgemm(m - i, BLOCK_SIZE, BLOCK_SIZE,
                        &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
        }
        naive_dgemm(m - i, BLOCK_SIZE, k - p,
                    &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
    }
    for (i = 0; i <= m - BLOCK_SIZE; i += BLOCK_SIZE) {
        for (p = 0; p <= k - BLOCK_SIZE; p += BLOCK_SIZE) {
            naive_dgemm(BLOCK_SIZE, n - j, BLOCK_SIZE,
                        &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
        }
        naive_dgemm(BLOCK_SIZE, n - j, k - p,
                    &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
    }
    {
        for (p = 0; p <= k - BLOCK_SIZE; p += BLOCK_SIZE) {
            naive_dgemm(m - i, n - j, BLOCK_SIZE,
                        &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
        }
        naive_dgemm(m - i, n - j, k - p,
                    &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
    }
}

inline static void naive_dgemm(int m, int n, int k, double *a, int lda, 
                               double *b, int ldb,
                               double *c, int ldc) {
    int j;
    for (j = 0; j <= n - 8; j += 8) {
        for (int p = 0; p < k; p++) {
            register
            double b_pj  = B(p, j),     b_pj1 = B(p, j + 1),
                   b_pj2 = B(p, j + 2), b_pj3 = B(p, j + 3),
                   b_pj4 = B(p, j + 4), b_pj5 = B(p, j + 5),
                   b_pj6 = B(p, j + 6), b_pj7 = B(p, j + 7);
            for (int i = 0; i < m; i++) {
                register double a_ip = A(i, p);
                C(i, j)     += a_ip * b_pj;
                C(i, j + 1) += a_ip * b_pj1;
                C(i, j + 2) += a_ip * b_pj2;
                C(i, j + 3) += a_ip * b_pj3;
                C(i, j + 4) += a_ip * b_pj4;
                C(i, j + 5) += a_ip * b_pj5;
                C(i, j + 6) += a_ip * b_pj6;
                C(i, j + 7) += a_ip * b_pj7;
            }
        }
    }
    for (; j < n; j++) {
        for (int p = 0; p < k; p++) {
            for (int i = 0; i < m; i++) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
}
