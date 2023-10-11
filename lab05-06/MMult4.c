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

// Used to pass arguments to child threads
struct Arg {
    int start;
    int end;
};

// These static variables are used to avoid cumbersome Args; should be set only once
static int _y, _lda, _ldb, _ldc, _k;
static double *_a, *_b, *_c;

static void full_block_mult(int k, double *a, int lda, 
                            double *b, int ldb,
                            double *c, int ldc);

static void block_mult(int m, int n, int k, double *a, int lda, 
                               double *b, int ldb,
                               double *c, int ldc);

static void block_mult_musk(struct Arg *args);

/* Routine for computing C = A * B + C */
/* (m*n) = (m*k) * (k*n) */
void MY_MMult(int m, int n, int k, double *a, int lda, 
                                   double *b, int ldb,
                                   double *c, int ldc) {
    int x, y;       // number of blocks in horizontal and verticle directions
    x = n / BLOCK_SIZE;
    y = m / BLOCK_SIZE;

    _y = y; _k = k; _lda = lda; _ldb = ldb; _ldc = ldc;
    _a = a; _b = b; _c = c;                     // Set static variables

    double *block_a, *block_b, *block_c;        // position of blocks

    pthread_t tid[NUM_THREADS];
    struct Arg args[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        int start = x * t / NUM_THREADS;
        int end = x * (t + 1) / NUM_THREADS;
        args[t].start = start;
        args[t].end = end;
        Pthread_create(&tid[t], NULL, (void *(*) (void *)) block_mult_musk, (void *) &args[t]);
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
        block_mult(BLOCK_SIZE, n - right_edge, k, block_a, lda, block_b, ldb,  block_c, ldc);
    }
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

#define _A(i,j) _a[ (j)*_lda + (i) ]
#define _B(i,j) _b[ (j)*_ldb + (i) ]
#define _C(i,j) _c[ (j)*_ldc + (i) ]

static void block_mult_musk(struct Arg *args) {
    int start = args->start;
    int end   = args->end;

    double *block_a, *block_b, *block_c;
    for (int j = start; j < end; j++) {
        for (int i = 0; i < _y; i++) {
            block_c = &_C(i * BLOCK_SIZE, j * BLOCK_SIZE);
            block_a = &_A(i * BLOCK_SIZE, 0);
            block_b = &_B(0, j * BLOCK_SIZE);
            full_block_mult(_k, block_a, _lda, block_b, _ldb, block_c, _ldc);
        }
    }
}
