#include <stdlib.h>
#include <malloc.h>
#include <mpi.h>
#include <cblas.h>
#include "dgemm_mpi.h"
#include "utilities.h"
#include "parameters.h"

int main(int argc, char **argv) {
    /* Initialize MPI */
    int my_id, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

    for (int size = S_FIRST; size <= S_LAST; size += S_STEP) {
        /* Matrices size */
        int m, n, k;
        m = n = k = size;
        int lda, ldb, ldc;
        lda = ldb = ldc = size;     // if lda == a, etc.
        /* Matrices */
        double *a, *b, *c, *c_ref;
        a = b = c = c_ref = NULL;
        /* Timer */
        double tic = 0.0;

        if (my_id == MAIN_PROCESS) {
            /* Allocate memory for matrices in main process */
            a = calloc(m * k, sizeof(double));
            b = calloc(k * n, sizeof(double));
            c = calloc(m * n, sizeof(double));
            c_ref = calloc(m * n, sizeof(double));
            if (!(a && b && c && c_ref)) {
                fprintf(stderr, "Calloc failed\n");
                exit(0);
            }
            /* Randomize matrices A, B, and C */
            fill_matrix(m, k, a, lda);
            fill_matrix(k, n, b, ldb);
            fill_matrix(m, n, c, ldc);
            /* Caculate reference matrix c_ref */
            copy_matrix(m, n, c_ref, ldc, c, ldc);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        (blasint) m, (blasint) n, (blasint) k, 1.0, a, lda, b, ldb, 1.0, c_ref, ldc);

            tic = MPI_Wtime();
        }

        mpi_dgemm(my_id, num_processes, m, n, k, a, lda, b, ldb, c, ldc);

        if (my_id == MAIN_PROCESS) {
            /* Show time elapsed, Gflops, and error value compared with c_ref */
            double time_elapsed = MPI_Wtime() - tic;
            double gflops = 2.0 * m * n * k / time_elapsed / 1e9;
            double error_value = compare_matrices(m, n, c, ldc, c_ref, ldc);
            printf("%d %le %le\n", size, gflops / time_elapsed, error_value);
            free(a); free(b); free(c); free(c_ref);
        }
    }

    MPI_Finalize();
    return 0;
}
