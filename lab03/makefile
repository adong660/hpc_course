OLD  := MMult0
NEW  := MMult_openblas
#
# sample makefile
#

CC         := gcc
LINKER     := $(CC)
CFLAGS     := -Wall
LDFLAGS    := -lm -I /opt/OpenBLAS/include/ -L /opt/OpenBLAS/lib/ -lopenblas

UTIL       := copy_matrix.o \
              compare_matrices.o \
              random_matrix.o \
              dclock.o \
              REF_MMult.o \
              print_matrix.o \
			  parameters.h

TEST       := test_MMult.o
OBJ        := $(wildcard *.o)

all:
	make $(OBJ)
	make test_MMult_old.x
	make test_MMult_new.x

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

MMult0.o: MMult0.c
	$(CC) $(CFLAGS) -c $< -o $@

MMult0_O1.o: MMult0.c
	$(CC) $(CFLAGS) -O1 -c $< -o $@

MMult0_O2.o: MMult0.c
	$(CC) $(CFLAGS) -O2 -c $< -o $@

MMult0_O3.o: MMult0.c
	$(CC) $(CFLAGS) -O3 -c $< -o $@

MMult0_O2_sse.o: MMult0.c
	$(CC) $(CFLAGS) -O2 -msse3 -c $< -o $@

MMult1.o: MMult1.c
	$(CC) $(CFLAGS) -c $< -o $@

MMult_openblas.o: MMult_openblas.c
	$(CC) $(CFLAGS) -I /opt/OpenBLAS/include/ -L /opt/OpenBLAS/lib/ -lopenblas -c $< -o $@

test_MMult_old.x: $(TEST) $(OLD).o $(UTIL)
	$(LINKER) $(TEST) $(OLD).o $(UTIL) $(LDFLAGS) $(BLAS_LIB) -o $@

test_MMult_new.x: $(TEST) $(NEW).o $(UTIL)
	$(LINKER) $(TEST) $(NEW).o $(UTIL) $(LDFLAGS) $(BLAS_LIB) -o $@

export OMP_NUM_THREADS=1
export GOTO_NUM_THREADS=1

run:
	make all

	echo "version = '$(OLD)';" > output_$(OLD).m
	echo $$OMP_NUM_THREADS
	./test_MMult_old.x >> output_$(OLD).m
	cp output_$(OLD).m output_old.m

	echo "version = '$(NEW)';" > output_$(NEW).m
	echo $$OMP_NUM_THREADS
	./test_MMult_new.x >> output_$(NEW).m
	cp output_$(NEW).m output_new.m

	python3 PlotAll.py

clean:
	rm -f *.o *~ core *.x

cleanall:
	rm -f *.o *~ core *.x output*.m *.eps *.png *.jpg
