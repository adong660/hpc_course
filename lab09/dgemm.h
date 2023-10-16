#ifndef DGEMM_H
#define DGEMM_H

void MY_MMult(int m, int n, int k, double *a, int lda, 
              double *b, int ldb, double *c, int ldc);

#endif
