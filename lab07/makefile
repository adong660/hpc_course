CC       := mpicc
LINKER   := $(CC)
CFLAGS   := -Wall -g -O1

OBJ      := test_dgemm_mpi.o dgemm_mpi.o utilities.o
EXEC     := test_dgemm_mpi

NUM_PROCESSES := 16

all:
	make test_dgemm_mpi

run:
	make all
	mpiexec -n $(NUM_PROCESSES) ./test_dgemm_mpi

test_dgemm_mpi: $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o test_dgemm_mpi

test_dgemm_mpi.o: test_dgemm_mpi.c dgemm_mpi.h utilities.h parameters.h

dgemm_mpi.o: dgemm_mpi.c dgemm_mpi.h

utilities.o: utilities.c utilities.h

clean:
	rm -f $(OBJ) $(EXEC)
