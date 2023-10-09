#include <stdlib.h>
#include <malloc.h>
#include <mpi.h>
#include "dgemm_mpi.h"

#define A(i, j) a[(i) + (j) * lda]
#define B(i, j) b[(i) + (j) * ldb]
#define C(i, j) c[(i) + (j) * ldc]

void mpi_dgemm(int my_id, int num_processes, int m, int n, int k, 
               double *a, int lda, double *b, int ldb, double *c, int ldc) {
    /* Width of the section this process should calculate */
    int my_width = n * (my_id + 1) / num_processes - n * my_id / num_processes;
    /* Start positions for each process used by the main process */
    int n_starts[num_processes + 1];
    if (my_id == MAIN_PROCESS) {
        for (int rank = 0; rank < num_processes; rank++) {
            int n_start = n * rank / num_processes;
            n_starts[rank] = n_start;
        }
        n_starts[num_processes] = n;
    }

    /* Allocate memory for matrices in child processes */
    if (my_id != MAIN_PROCESS) {
        a = calloc(m * k, sizeof(double));
        b = calloc(k * my_width, sizeof(double));
        c = calloc(m * my_width, sizeof(double));
        if (!(a && b && c)) {
            fprintf(stderr, "Calloc failed\n");
            exit(0);
        }
    }

    /* Broadcast matrix A to child processes */
    MPI_Bcast(a, m * k, MPI_DOUBLE, MAIN_PROCESS, MPI_COMM_WORLD);
    /* Send sections of matrices B, and C to child processes */
    if (my_id == MAIN_PROCESS) {
        for (int rank = 1; rank < num_processes; rank++) {
            int section_width = n_starts[rank + 1] - n_starts[rank];
            MPI_Send(&B(0, n_starts[rank]), k * section_width,
                     MPI_DOUBLE, rank, 10, MPI_COMM_WORLD);
            MPI_Send(&C(0, n_starts[rank]), m * section_width,
                     MPI_DOUBLE, rank, 11, MPI_COMM_WORLD);
        }
    } else {
        MPI_Status status;
        MPI_Recv(b, k * my_width, MPI_DOUBLE,
                 MAIN_PROCESS, 10, MPI_COMM_WORLD, &status);
        MPI_Recv(c, m * my_width, MPI_DOUBLE,
                 MAIN_PROCESS, 11, MPI_COMM_WORLD, &status);
    }

    /* Calculate each process's own section */
    naive_dgemm(m, my_width, k, a, lda, b, ldb, c, ldc);

    /* Main process collects calculation result to matrix C */
    if (my_id == MAIN_PROCESS) {
        for (int rank = 1; rank < num_processes; rank++) {
            int buff_count = m * (n_starts[rank + 1] - n_starts[rank]);
            MPI_Status status;
            MPI_Recv(&C(0, n_starts[rank]), buff_count, MPI_DOUBLE,
                     rank, 12, MPI_COMM_WORLD, &status);
        }
    } else {
        int buff_count = m * my_width;     // Needs to be corrected in case that ldc != m
        MPI_Send(c, buff_count, MPI_DOUBLE, MAIN_PROCESS, 12, MPI_COMM_WORLD);
        free(a); free(b); free(c);
    }
}

/* Naive implementation of matrix multiplication */
void naive_dgemm(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc) {
    for (int j = 0; j < n; j++) {
        for (int p = 0; p < k; p++) {
            for (int i = 0; i < m; i++) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
}
