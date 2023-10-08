#ifndef UTILITIES_H
#define UTILITIES_H

#define A(i, j) a[(i) + (j) * lda]
#define B(i, j) b[(i) + (j) * ldb]
#define C(i, j) c[(i) + (j) * ldc]

void copy_matrix(int m, int n, double *dst, int ld_dst, const double *rsc, int ld_rsc);

void print_matrix(int m, int n, double *matrix, int ld);

void fill_matrix(int m, int n, double *matrix, int ld);

double compare_matrices(int m, int n, double *a, int lda, double *b, int ldb);

#endif
