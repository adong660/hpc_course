/* OpenMP implementation of blocked matrix multiplication */
/* Split the matrix into fixed size blocks */
#include <omp.h>
#include "MMult.h"

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

#define BLOCK_SIZE 32
#define NUM_THREADS 16

static void full_block_mult(int k, double *a, int lda, 
                            double *b, int ldb,
                            double *c, int ldc );

static void block_mult(int m, int n, int k, double *a, int lda, 
                               double *b, int ldb,
                               double *c, int ldc);

/* Routine for computing C = A * B + C */
/* (m*n) = (m*k) * (k*n) */
void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc ) {
    omp_set_num_threads(NUM_THREADS);
    int x, y;       // number of blocks in horizontal and verticle directions
    x = n / BLOCK_SIZE;
    y = m / BLOCK_SIZE;

    double *block_a, *block_b, *block_c;        // position of blocks

#pragma omp parallel for
    for (int j = 0; j < x; j++) {
        for (int i = 0; i < y; i++) {
            block_c = &C(i * BLOCK_SIZE, j * BLOCK_SIZE);
            block_a = &A(i * BLOCK_SIZE, 0);
            block_b = &B(0, j * BLOCK_SIZE);
            full_block_mult(k, block_a, lda, block_b, ldb, block_c, ldc);
        }
    }

    int right_edge, down_edge;
    right_edge = x * BLOCK_SIZE;
    down_edge  = y * BLOCK_SIZE;
#pragma omp parallel for
    for (int i = 0; i < y; i++) {
        block_c = &C(i * BLOCK_SIZE, right_edge);
        block_a = &A(i * BLOCK_SIZE, 0);
        block_b = &B(0, right_edge);
        block_mult(BLOCK_SIZE, n - right_edge, k, block_a, lda, block_b, ldb,  block_c, ldc);
    }
#pragma omp parallel for
    for (int j = 0; j < x; j++) {
        block_c = &C(down_edge, j * BLOCK_SIZE);
        block_a = &A(down_edge, 0);
        block_b = &B(0, j * BLOCK_SIZE);
        block_mult(m - down_edge, BLOCK_SIZE, k, block_a, lda, block_b, ldb,  block_c, ldc);
    }
    {
        block_c = &C(down_edge, right_edge);
        block_a = &A(down_edge, 0);
        block_b = &B(0, right_edge);
        block_mult(m - down_edge, n - right_edge, k, block_a, lda, block_b, ldb, block_c, ldc);
    }
}

static void full_block_mult(int k, double *a, int lda, 
                            double *b, int ldb,
                            double *c, int ldc ) {
    int i, j, p;
    for (j = 0; j < BLOCK_SIZE; j++) {
        for (p = 0; p < k; p++) {
            for (i = 0; i < BLOCK_SIZE; i++) {
                C(i, j) = C(i, j) + A(i, p) * B(p, j);
            }
        }
    }
}

static void block_mult(int m, int n, int k,
                              double *a, int lda, 
                              double *b, int ldb,
                              double *c, int ldc ) {
    int i, j, p;
    for (j = 0; j < n; j++) {
        for (p = 0; p < k; p++) {
            for (i = 0; i < m; i++) {
                C(i, j) = C(i, j) + A(i, p) * B(p, j);
            }
        }
    }
}
