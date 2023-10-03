#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "common.h"

int main(int argc, char *argv[])
{
    printf("hello world (pid:%d)\n", (int) getpid());
    int rc = fork();
    if (rc < 0) {           // fork failed; exit
        fprintf(stderr, "fork failed\n");
        exit(1);
    } else if (rc == 0) {   // child
        printf("hello, I am child (pid:%d)\n", (int) getpid());
        Spin(1);
        printf("end of process (pid:%d)\n", (int) getpid());
    } else {                // parent
        printf("hello, I am parent of %d (pid:%d)\n",
	        rc, (int) getpid());
        Spin(2);
        printf("end of process (pid:%d)\n", (int) getpid());
    }
    return 0;
}
