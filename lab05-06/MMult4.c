/* Pthreads implementation of blocked matrix multiplication */
/* Split the matrix into fixed size blocks */
#include "MMult.h"
#include <pthread.h>
#include "common_threads.h"

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

#define BLOCK_SIZE 32
#define NUM_THREADS 16

struct Arg {
    int start, end, y;
    int lda, ldb, ldc, k;
    double *a, *b, *c;
};

static void full_block_mult(int k, double *a, int lda, 
                            double *b, int ldb,
                            double *c, int ldc );

static void notfull_block_mult(int m, int n, int k, double *a, int lda, 
                              double *b, int ldb,
                              double *c, int ldc );

static void thread_one(void *args);

/* Routine for computing C = A * B + C */
/* (m*n) = (m*k) * (k*n) */
void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc ) {
    int x, y;       // number of blocks in horizontal and verticle directions
    x = n / BLOCK_SIZE;
    y = m / BLOCK_SIZE;

    double *block_a, *block_b, *block_c;        // position of blocks

    pthread_t tid[NUM_THREADS];
    struct Arg args[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        int start = x * t / NUM_THREADS;
        int end = x * (t + 1) / NUM_THREADS;
        struct Arg arg = {start, end, y, lda, ldb, ldc, k, a, b, c};
        args[t] = arg;
        Pthread_create(&tid[t], NULL, (void *(*) (void *)) thread_one, (void *) &args[t]);
    }
    for (int t = 0; t < NUM_THREADS; t++) {
        Pthread_join(tid[t], NULL);
    }

    int right_edge, down_edge;
    right_edge = x * BLOCK_SIZE;
    down_edge  = y * BLOCK_SIZE;
    for (int i = 0; i < y; i++) {
        block_c = &C(i * BLOCK_SIZE, right_edge);
        block_a = &A(i * BLOCK_SIZE, 0);
        block_b = &B(0, right_edge);
        notfull_block_mult(BLOCK_SIZE, n - right_edge, k, block_a, lda, block_b, ldb,  block_c, ldc);
    }
    for (int j = 0; j < x; j++) {
        block_c = &C(down_edge, j * BLOCK_SIZE);
        block_a = &A(down_edge, 0);
        block_b = &B(0, j * BLOCK_SIZE);
        notfull_block_mult(m - down_edge, BLOCK_SIZE, k, block_a, lda, block_b, ldb,  block_c, ldc);
    }
    {
        block_c = &C(down_edge, right_edge);
        block_a = &A(down_edge, 0);
        block_b = &B(0, right_edge);
        notfull_block_mult(m - down_edge, n - right_edge, k, block_a, lda, block_b, ldb, block_c, ldc);
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

static void notfull_block_mult(int m, int n, int k,
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

static void thread_one(void *args) {
    struct Arg *the_args = (struct Arg *) args;
    int start = the_args->start;
    int end = the_args->end;
    int y = the_args->y;
    double *a = the_args->a;
    double *b = the_args->b;
    double *c = the_args->c;
    int lda = the_args->lda;
    int ldb = the_args->ldb;
    int ldc = the_args->ldc;
    int k = the_args->k;

    double *block_a, *block_b, *block_c;
    for (int j = start; j < end; j++) {
        for (int i = 0; i < y; i++) {
            block_c = &C(i * BLOCK_SIZE, j * BLOCK_SIZE);
            block_a = &A(i * BLOCK_SIZE, 0);
            block_b = &B(0, j * BLOCK_SIZE);
            full_block_mult(k, block_a, lda, block_b, ldb, block_c, ldc);
        }
    }
}
