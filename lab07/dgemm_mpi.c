#include <stdlib.h>
#include <malloc.h>
#include <mpi.h>
#include "utilities.h"

#define MATRIX_SIZE 2000

void naive_dgemm(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);

static void allocate_matrices(int id, int a_size, int b_size, int c_size,
                              double **a, double **b, double **c, double **c_ref);

static void free_memory(int id, double *a, double *b, double *c, double *c_ref);

int main(int argc, char **argv) {

    /* Initialize MPI */
    int my_id, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

    double time_elapsed = 0.0, tic = 0.0;   // Timer
    tic = MPI_Wtime();

    /* Matrices size */
    int m, n, k;
    m = n = k = MATRIX_SIZE;      // Test matrix size   TODO
    int lda, ldb, ldc;
    lda = ldb = ldc = m;          // if lda == a, etc.
    int a_size, b_size, c_size;
    a_size = m * k;
    b_size = k * n;
    c_size = m * n;

    /* Allocate memory for matrices */
    double *a, *b, *c, *c_ref;
    a = b = c = c_ref = NULL;
    allocate_matrices(my_id, a_size, b_size, c_size, &a, &b, &c, &c_ref);

    time_elapsed += MPI_Wtime() - tic;

    /* Randomize matrices in process 0 */
    if (my_id == 0) {
        fill_matrix(m, k, a, lda);
        fill_matrix(k, n, b, ldb);
        fill_matrix(m, n, c, ldc);
    }

    /* Caculate reference matrix c_ref */
    if (my_id == 0) {
        copy_matrix(m, n, c_ref, ldc, c, ldc);
        naive_dgemm(m, n, k, a, lda, b, ldb, c_ref, ldc);
    }

    tic = MPI_Wtime();

    /* Then broadcast matrices A, B, and C to other processes */
    MPI_Bcast(a, a_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, b_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(c, c_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /* Calculate each process's own part */
    int n_start = n * my_id / num_processes;
    int n_end = n * (my_id + 1) / num_processes;
    naive_dgemm(m, n_end - n_start, k, &A(0, 0), lda, &B(0, n_start), ldb, &C(0, n_start), ldc);

    /* Process 0 collects data sent by other processes */
    #define TAG 20      // Message tag
    if (my_id > 0) {    // Send
        int buff_count = (n_end - n_start) * m;     // Needs to be corrected in case that ldc != m
        MPI_Send(&C(0, n_start), buff_count, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD);
    }
    else {              // Collect
        for (int rank = 1; rank < num_processes; rank++) {
            int n_start = n * rank / num_processes;
            int n_end = n * (rank + 1) / num_processes;
            int buff_count = (n_end - n_start) * m;
            MPI_Status status;
            MPI_Recv(&C(0, n_start), buff_count, MPI_DOUBLE, rank, TAG, MPI_COMM_WORLD, &status);
        }
        /* Time elapsed and error value */
        time_elapsed = MPI_Wtime() - tic;
        double gflops = 2.0 * m * n * k / time_elapsed / 1e9;
        double error_value = compare_matrices(m, n, c, ldc, c_ref, ldc);
        printf("Time elapsed: %f\n", time_elapsed);
        printf("Gflops: %f\n", gflops);
        printf("Error value: %f\n", error_value);
    }
    free_memory(my_id, a, b, c, c_ref);

    MPI_Finalize();
    return 0;
}

/* Naive implementation of matrix multiplication,
   also used for multi-process implementation   */
void naive_dgemm(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc) {
    for (int j = 0; j < n; j++) {
        for (int p = 0; p < k; p++) {
            for (int i = 0; i < m; i++) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
}

/* Allocate memory for each matrix in each process */
static void allocate_matrices(int id, int a_size, int b_size, int c_size,
                              double **a, double **b, double **c, double **c_ref) {
    *a = calloc(a_size, sizeof(double));
    *b = calloc(b_size, sizeof(double));
    *c = calloc(c_size, sizeof(double));
    if (!(*a && *b && *c)) {
        fprintf(stderr, "calloc failed\n");
        exit(0);
    }
    /* Only process 0 needs reference matrix C_ref */
    if (id == 0) {
        *c_ref = calloc(c_size, sizeof(double));
        if (!(*c_ref)) {
            fprintf(stderr, "calloc failed\n");
            exit(0);
        }
    }
}

static void free_memory(int id, double *a, double *b, double *c, double *c_ref) {
    free(a);
    free(b);
    free(c);
    if (id == 0) {
        free(c_ref);
    }
}
