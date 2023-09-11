#include <stdio.h>
#include "cblas.h"
#include "dgemm.h"

// Compute: C = alpha*A*B + beta*C

void dgemm_naive(int m, int n, int k, double alpha, double beta,
           double A[m][k], double B[k][n], double C[m][n]) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            C[i][j] *= beta;
            for (int p = 0; p < k; p++)
                C[i][j] += alpha * A[i][p] * B[p][j];
        }
    return;
}

void dgemm_cblas(int m, int n, int k, double alpha, double beta,
           double A[m][k], double B[k][n], double C[m][n]) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
    return;
}
