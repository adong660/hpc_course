#include "utils.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "dgemm.h"

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

void copy_matrix(int m, int n, double *a, int lda, double *b, int ldb) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            B(i, j) = A(i, j);
}

void random_matrix(int m, int n, double *a, int lda) {
    srand48((unsigned int) time(NULL));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            A(i, j) = drand48();
}

double compare_matrices(int m, int n, double *a, int lda, double *b, int ldb) {
    double max_diff = 0.0, diff;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            diff = fabs(A(i, j) - B(i, j));
            max_diff = diff > max_diff ? diff : max_diff;
        }
    return max_diff;
}

void REF_MMult(int m, int n, int k, double *a, int lda, double *b, int ldb, double *cref, int ldc) {
    MY_MMult(m, n, k, a, lda, b, ldb, cref, ldc);
}

double dclock() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1.0e-6;
}
