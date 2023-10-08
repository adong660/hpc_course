#include <stdio.h>
#include <stdlib.h>

#include "parameters.h"
#include "MMult.h"

void copy_matrix(int, int, double *, int, double *, int);
void random_matrix(int, int, double *, int);

double dclock();

int main()
{
    int
        p,
        m, n, k,
        lda, ldb, ldc,
        rep;

    double
        dtime,
        dtime_best,
        gflops;

    double
        *a,
        *b, *c, *cold;

    printf("MY_MMult = [\n");

    for (p = PFIRST; p <= PLAST; p += PINC)
    {
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

        /* Generate random matrices A, B, Cold */
        random_matrix(m, k, a, lda);
        random_matrix(k, n, b, ldb);
        random_matrix(m, n, cold, ldc);

        /* Time the "optimized" implementation */
        for (rep = 0; rep < NREPEATS; rep++)
        {
            copy_matrix(m, n, cold, ldc, c, ldc);
            dtime = dclock();
            MY_MMult(m, n, k, a, lda, b, ldb, c, ldc);
            dtime = dclock() - dtime;

            if (rep == 0)
                dtime_best = dtime;
            else
                dtime_best = (dtime < dtime_best ? dtime : dtime_best);
        }

        printf("%d %le %le\n", p, gflops / dtime_best, 0.0);
        fflush(stdout);

        free(a);
        free(b);
        free(c);
        free(cold);
    }

    printf("];\n");

    exit(0);
}
