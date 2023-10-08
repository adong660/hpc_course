#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "utilities.h"

#define dst(i, j) dst[(i) + (j) * ld_dst]
#define rsc(i, j) rsc[(i) + (j) * ld_rsc]
void copy_matrix(int m, int n, double *dst, int ld_dst, const double *rsc, int ld_rsc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dst(i, j) = rsc(i, j);
        }
    }
}

#define matrix(i, j) matrix[(i) + (j) * ld]
void print_matrix(int m, int n, double *matrix, int ld) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f\t", matrix(i, j));
        }
        printf("\n");
    }
}

void fill_matrix(int m, int n, double *matrix, int ld) {
    srand48((unsigned int) time(NULL));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix(i, j) = drand48();
        }
    }
}

double compare_matrices(int m, int n, double *a, int lda, double *b, int ldb) {
    double max_diff = 0.0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            double diff = fabs(A(i, j) - B(i, j));
            max_diff = (diff > max_diff ? diff : max_diff);
        }
    return max_diff;
}
