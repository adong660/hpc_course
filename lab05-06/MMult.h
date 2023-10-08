/* The common API for all matrix multiplications */
#ifndef MMULT_H
#define MMULT_H

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc );

#endif
