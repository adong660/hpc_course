#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "parameters.h"
#include "dgemm.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "usage: test_MMult.c <--test|--time>\n");
        exit(0);
    }

    bool test;
    if (strcmp(argv[1], "--test") == 0)
        test = true;
    else if (strcmp(argv[1], "--time") == 0)
        test = false;
    else {
        fprintf(stderr, "usage: test_MMult.c <--test|--time>\n");
        exit(0);
    }

    int p, m, n, k, lda, ldb, ldc;
    double dtime, gflops, diff;
    double *a, *b, *c, *cref, *cold;

    printf("MY_MMult = [\n");

    for (p = PFIRST; p <= PLAST; p += PINC) {
        m = (M == -1 ? p : M);
        n = (N == -1 ? p : N);
        k = (K == -1 ? p : K);

        gflops = 2.0 * m * n * k * 1.0e-09;

        lda = (LDA == -1 ? m : LDA);
        ldb = (LDB == -1 ? k : LDB);
        ldc = (LDC == -1 ? m : LDC);

        /* Allocate space for the matrices */
        /* Note: I create an extra column in A to make sure that
           prefetching beyond the matrix does not cause a segfault */
        a = (double *)malloc(lda * (k + 1) * sizeof(double));
        b = (double *)malloc(ldb * n * sizeof(double));
        c = (double *)malloc(ldc * n * sizeof(double));
        cold = (double *)malloc(ldc * n * sizeof(double));
        cref = (double *)malloc(ldc * n * sizeof(double));

        /* Generate random matrices A, B, Cold */
        random_matrix(m, k, a, lda);
        random_matrix(k, n, b, ldb);
        random_matrix(m, n, cold, ldc);

        if (test) {
            /* Run the reference implementation so the answers can be compared */
            copy_matrix(m, n, cold, ldc, cref, ldc);
            REF_MMult(m, n, k, a, lda, b, ldb, cref, ldc);
        }

        /* Time the "optimized" implementation */
        copy_matrix(m, n, cold, ldc, c, ldc);
        dtime = dclock();
        MY_MMult(m, n, k, a, lda, b, ldb, c, ldc);
        dtime = dclock() - dtime;

        diff = test ? compare_matrices(m, n, c, ldc, cref, ldc) : -1.0;

        printf("%d %le %le\n", p, gflops / dtime, diff);
        fflush(stdout);

        free(a); free(b); free(c); free(cold); free(cref);
    }

    printf("];\n");

    return 0;
}
