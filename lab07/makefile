CC       := mpicc
LINKER   := $(CC)
CFLAGS   := -Wall -g -O1 -I /opt/OpenBLAS/include/
LDFLAGS  := -L /opt/OpenBLAS/lib/ -lopenblas

OBJ      := test_dgemm_mpi.o dgemm_mpi.o utilities.o
EXEC     := test_dgemm_mpi

NUM_PROCESSES := 16

all:
	make test_dgemm_mpi

run:
	make all

	echo "version = 'dgemm_mpi';" > output_dgemm_mpi.m
	echo "MY_MMult = [" >> output_dgemm_mpi.m
	mpiexec -n $(NUM_PROCESSES) ./test_dgemm_mpi >> output_dgemm_mpi.m
	echo "];" >> output_dgemm_mpi.m
	cp output_dgemm_mpi.m output_new.m

	python3 PlotAll.py

test_dgemm_mpi: $(OBJ)
	$(CC) $(OBJ) -o test_dgemm_mpi $(LDFLAGS)

test_dgemm_mpi.o: test_dgemm_mpi.c dgemm_mpi.h utilities.h parameters.h

dgemm_mpi.o: dgemm_mpi.c dgemm_mpi.h

utilities.o: utilities.c utilities.h

clean:
	rm -f $(OBJ) $(EXEC)

cleanall:
	clean
	rm -f *.m *.png
