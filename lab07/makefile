CC       := mpicc
LINKER   := $(CC)
CFLAGS   := -Wall -g -O1

OBJ      := dgemm_mpi.o utilities.o
NUM_PROCESS := 16

all:
	make dgemm_mpi

run:
	make all
	mpiexec -n $(NUM_PROCESS) ./dgemm_mpi

dgemm_mpi: $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o dgemm_mpi
