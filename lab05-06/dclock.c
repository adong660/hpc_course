#include <sys/time.h>
#include <time.h>

/* Adapted from the bl2_clock() routine in the BLIS library */

double dclock()
{
        double         the_time;
        struct timeval tv;

        gettimeofday( &tv, NULL );

        the_time = tv.tv_sec + tv.tv_usec * 1.0e-6;

        return the_time;
}

