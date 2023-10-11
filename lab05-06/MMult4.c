/* Pthreads implementation of multi-threaded block matrix multiplication */
/* Split the matrix into fixed size blocks */
#include "MMult.h"
#include <pthread.h>
#include "common_threads.h"

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

#define BLOCK_SIZE 20
#define NUM_THREADS 16

// Used to pass arguments to child threads
struct Arg {
    int m, n, k;
    int lda, ldb, ldc;
    double *a, *b, *c;
};

static void block_dgemm(int m, int n, int k, double *a, int lda, 
                                      double *b, int ldb,
                                      double *c, int ldc);

static inline void naive_dgemm(int m, int n, int k, double *a, int lda, 
                                      double *b, int ldb,
                                      double *c, int ldc);

static void *dgemm_thread(void *arg);

/* Routine for computing C = A * B + C */
/* (m*n) = (m*k) * (k*n) */
void MY_MMult(int m, int n, int k, double *a, int lda, 
                                   double *b, int ldb,
                                   double *c, int ldc) {
    pthread_t tid[NUM_THREADS];
    struct Arg arg[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        int n_start = t * n / NUM_THREADS;
        int n_end = (t + 1) * n / NUM_THREADS;
        arg[t].m = m; arg[t].n = n_end - n_start; arg[t].k = k;
        arg[t].a = a; arg[t].b = &B(0, n_start); arg[t].c = &C(0, n_start);
        arg[t].lda = lda; arg[t].ldb = ldb; arg[t].ldc = ldc;
        Pthread_create(&tid[t], NULL, dgemm_thread, (void *) &arg[t]);
    }
    for (int t = 0; t < NUM_THREADS; t++) {
        Pthread_join(tid[t], NULL);
    }
}

static void *dgemm_thread(void *_arg) {
    struct Arg *arg = _arg;
    int m, n, k;
    int lda, ldb, ldc;
    double *a, *b, *c;
    m = arg->m; n = arg->n; k = arg->k;
    lda = arg->lda; ldb = arg->ldb; ldc = arg->ldc;
    a = arg->a; b = arg->b; c = arg->c;
    block_dgemm(m, n, k, a, lda, b, ldb, c, ldc);
    return NULL;
}

static void block_dgemm(int m, int n, int k, double *a, int lda, 
                        double *b, int ldb,
                        double *c, int ldc) {
    int i = 0, j = 0, p = 0;
    for (i = 0; i <= m - BLOCK_SIZE; i += BLOCK_SIZE) {
        for (j = 0; j <= n - BLOCK_SIZE; j += BLOCK_SIZE) {
            for (p = 0; p <= k - BLOCK_SIZE; p += BLOCK_SIZE) {
                naive_dgemm(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE,
                            &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
            }
            naive_dgemm(BLOCK_SIZE, BLOCK_SIZE, k - p,
                        &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
        }
    }
    for (j = 0; j <= n - BLOCK_SIZE; j += BLOCK_SIZE) {
        for (p = 0; p <= k - BLOCK_SIZE; p += BLOCK_SIZE) {
            naive_dgemm(m - i, BLOCK_SIZE, BLOCK_SIZE,
                        &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
        }
        naive_dgemm(m - i, BLOCK_SIZE, k - p,
                    &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
    }
    for (i = 0; i <= m - BLOCK_SIZE; i += BLOCK_SIZE) {
        for (p = 0; p <= k - BLOCK_SIZE; p += BLOCK_SIZE) {
            naive_dgemm(BLOCK_SIZE, n - j, BLOCK_SIZE,
                        &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
        }
        naive_dgemm(BLOCK_SIZE, n - j, k - p,
                    &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
    }
    {
        for (p = 0; p <= k - BLOCK_SIZE; p += BLOCK_SIZE) {
            naive_dgemm(m - i, n - j, BLOCK_SIZE,
                        &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
        }
        naive_dgemm(m - i, n - j, k - p,
                    &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
    }
}

static inline void naive_dgemm(int m, int n, int k, double *a, int lda, 
                               double *b, int ldb,
                               double *c, int ldc) {
    for (int j = 0; j < n; j++) {
        for (int p = 0; p < k; p++) {
            for (int i = 0; i < m; i++) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
}
