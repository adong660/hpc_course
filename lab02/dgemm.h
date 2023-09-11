#ifndef DGEMM_H
#define DGEMM_H

void dgemm_naive(int m, int n, int k, double alpha, double beta,
           double A[m][k], double B[k][n], double C[m][n]);
void dgemm_cblas(int m, int n, int k, double alpha, double beta,
           double A[m][k], double B[k][n], double C[m][n]);

#endif
