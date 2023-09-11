#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "dgemm.h"

double time_dgemm(void (*dgemm)(int, int, int, double, double, double*, double*, double*),
                  int m, double *arrA, double *arrB, double *arrC) {
    const double alpha = 1.2;
    const double beta  = 2.6;
    struct timeval tic, toc;
    gettimeofday(&tic, NULL);
    dgemm(m, m, m, alpha, beta, arrA, arrB, arrC);
    gettimeofday(&toc, NULL);
    double elapsed_time = toc.tv_sec - tic.tv_sec + (toc.tv_usec - tic.tv_usec) / 1.0e6;
    return elapsed_time;
}

void update_records(int m, double duration, char *comment) {
    FILE *file = fopen("timeDGEMM.txt", "a");
    if (!file) {
        fprintf(stderr, "Failed to open timeDGEMM.txt\n");
        exit(1);
    }
    double gflops = (2l*m*m*m) / duration / 1.0e9;
    fprintf(file, "%4d x %4d\ttime used: %lfs\tGflops: %lf\t%s\n",
            m, m, duration, gflops, comment);
    fclose(file);
    return;
}

int main(int argc, char* argv[]) {
    if (argc != 2 || argv[1] <= 0) {
        fprintf(stderr, "Wrong number of arguments\n");
        exit(1);
    }
    int m = atoi(argv[1]);
    if (m <= 0) {
        fprintf(stderr, "Wrong value of argument\n");
        exit(1);
    }

    long size = m * m;
    double *arrA = malloc(sizeof(double) * size);
    double *arrB = malloc(sizeof(double) * size);
    double *arrC = malloc(sizeof(double) * size);
    if (!arrA || !arrB || !arrC) {
        fprintf(stderr, "Malloc failed\n");
        exit(1);
    }
    for (long i = 0; i < size; i++) {
        *(arrA + i) = *(arrB + i) = *(arrC + i) = 1.1;
    }
    double time_naive = time_dgemm(dgemm_naive, m, arrA, arrB, arrC);
    update_records(m, time_naive, "naive");
    double time_cblas = time_dgemm(dgemm_cblas, m, arrA, arrB, arrC);
    update_records(m, time_cblas, "cblas");

    free(arrA);
    free(arrB);
    free(arrC);

    return 0;
}
