/* Block + jpi naive + OpenMP */
#include "dgemm.h"

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

#define BLOCK_SIZE 20

static void block_dgemm(int m, int n, int k, double *a, int lda, 
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

static void block_dgemm(int m, int n, int k, double *a, int lda, 
                        double *b, int ldb,
                        double *c, int ldc) {
    #pragma omp parallel for num_threads(16)
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
