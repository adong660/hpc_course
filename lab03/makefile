OLD  := MMult0_O2
NEW  := MMult_openblas

CC         := gcc
LINKER     := $(CC)
CFLAGS     := -Wall -g
LDFLAGS    := -lm
BLAS_LIB   := -L /opt/OpenBLAS/lib/ -lopenblas

UTILS      := copy_matrix.o \
              compare_matrices.o \
              random_matrix.o \
              dclock.o \
              REF_MMult.o \
              print_matrix.o

TEST       := test_MMult.o
TEST_OBJ   := MMult0.o MMult0_O1.o MMult0_O2.o MMult0_O2_sse.o MMult0_O3.o MMult1.o MMult_openblas.o


all:
	make $(UTILS)
	make $(TEST_OBJ)
	rm -f *.x
	make test_MMult_old.x
	make test_MMult_new.x


MMult0_O1.o: MMult0.c
	$(CC) $(CFLAGS) -O1 -c $< -o $@

MMult0_O2.o: MMult0.c
	$(CC) $(CFLAGS) -O2 -c $< -o $@

MMult0_O3.o: MMult0.c
	$(CC) $(CFLAGS) -O3 -c $< -o $@

MMult0_O2_sse.o: MMult0.c
	$(CC) $(CFLAGS) -O2 -msse3 -c $< -o $@

MMult_openblas.o: MMult_openblas.c
	$(CC) $(CFLAGS) -I /opt/OpenBLAS/include/ -c $< -o $@

test_MMult.o: test_MMult.c parameters.h

test_MMult_old.x: $(TEST) $(OLD).o $(UTILS)
	$(LINKER) $(TEST) $(OLD).o $(UTILS) $(LDFLAGS) $(BLAS_LIB) -o $@

test_MMult_new.x: $(TEST) $(NEW).o $(UTILS)
	$(LINKER) $(TEST) $(NEW).o $(UTILS) $(LDFLAGS) $(BLAS_LIB) -o $@

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
