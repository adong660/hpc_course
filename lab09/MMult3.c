/* Block + jpi naive */
#include "dgemm.h"

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

#define BLOCK_SIZE 50

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
    for (int j = 0; j < n; j++) {
        for (int p = 0; p < k; p++) {
            for (int i = 0; i < m; i++) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
}

