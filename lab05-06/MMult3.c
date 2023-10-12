/* Single thread implementation of block matrix multiplication */
/* Split the matrix into fixed size blocks */
#include "MMult.h"

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

#define BLOCK_SIZE 20

/* Routine for computing C = A * B + C */
/* (m*n) = (m*k) * (k*n) */
void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc ) {
    int i = 0, j = 0, p = 0;
    for (i = 0; i <= m - BLOCK_SIZE; i += BLOCK_SIZE) {
        for (j = 0; j <= n - BLOCK_SIZE; j += BLOCK_SIZE) {
            for (p = 0; p <= k - BLOCK_SIZE; p += BLOCK_SIZE) {
                for (int jj = j; jj < j + BLOCK_SIZE; jj++) {
                    for (int pp = p; pp < p + BLOCK_SIZE; pp++) {
                        for (int ii = i; ii < i + BLOCK_SIZE; ii++) {
                            C(ii, jj) += A(ii, pp) * B(pp, jj);
                        }
                    }
                }
            }
            for (int jj = j; jj < j + BLOCK_SIZE; jj++) {
                for (int pp = p; pp < k; pp++) {
                    for (int ii = i; ii < i + BLOCK_SIZE; ii++) {
                        C(ii, jj) += A(ii, pp) * B(pp, jj);
                    }
                }
            }
        }
    }
    for (j = 0; j <= n - BLOCK_SIZE; j += BLOCK_SIZE) {
        for (p = 0; p <= k - BLOCK_SIZE; p += BLOCK_SIZE) {
            for (int jj = j; jj < j + BLOCK_SIZE; jj++) {
                for (int pp = p; pp < p + BLOCK_SIZE; pp++) {
                    for (int ii = i; ii < m; ii++) {
                        C(ii, jj) += A(ii, pp) * B(pp, jj);
                    }
                }
            }
        }
        for (int jj = j; jj < j + BLOCK_SIZE; jj++) {
            for (int pp = p; pp < k; pp++) {
                for (int ii = i; ii < m; ii++) {
                    C(ii, jj) += A(ii, pp) * B(pp, jj);
                }
            }
        }
    }
    for (i = 0; i <= m - BLOCK_SIZE; i += BLOCK_SIZE) {
        for (p = 0; p <= k - BLOCK_SIZE; p += BLOCK_SIZE) {
            for (int jj = j; jj < n; jj++) {
                for (int pp = p; pp < p + BLOCK_SIZE; pp++) {
                    for (int ii = i; ii < i + BLOCK_SIZE; ii++) {
                        C(ii, jj) += A(ii, pp) * B(pp, jj);
                    }
                }
            }
        }
        for (int jj = j; jj < n; jj++) {
            for (int pp = p; pp < k; pp++) {
                for (int ii = i; ii < i + BLOCK_SIZE; ii++) {
                    C(ii, jj) += A(ii, pp) * B(pp, jj);
                }
            }
        }
    }
    for (p = 0; p <= k - BLOCK_SIZE; p += BLOCK_SIZE) {
        for (int jj = j; jj < n; jj++) {
            for (int pp = p; pp < p + BLOCK_SIZE; pp++) {
                for (int ii = i; ii < m; ii++) {
                    C(ii, jj) += A(ii, pp) * B(pp, jj);
                }
            }
        }
    }
    for (int jj = j; jj < n; jj++) {
        for (int pp = p; pp < k; pp++) {
            for (int ii = i; ii < m; ii++) {
                    C(ii, jj) += A(ii, pp) * B(pp, jj);
            }
        }
    }
}
