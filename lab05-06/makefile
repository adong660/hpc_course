OLD  := MMult2
NEW  := MMult3

CC         := gcc
LINKER     := $(CC)
CFLAGS     := -Wall -g -O1
LDFLAGS    := -lm -L /opt/OpenBLAS/lib/ -lopenblas -fopenmp -pthread

UTILS      := copy_matrix.o \
              compare_matrices.o \
              random_matrix.o \
              dclock.o \
              REF_MMult.o \
              print_matrix.o \
			  parameters.h
TEST       := test_MMult.o
TIME       := time_MMult.o
TEST_OBJ   := MMult2.o MMult3.o MMult4.o MMult5.o MMult_openblas.o
EXECUTABLE := test_MMult_new.x test_MMult_old.x time_MMult_new.x time_MMult_old.x


all:
	make $(UTILS)
	make $(TEST_OBJ)
	rm -f *.x
	make $(EXECUTABLE)

MMult2.o: MMult2.c
	$(CC) $(CFLAGS) -c $< -o $@

MMult3.o: MMult3.c
	$(CC) $(CFLAGS) -c $< -o $@

MMult4.o: MMult4.c
	$(CC) $(CFLAGS) -pthread -c $< -o $@

MMult5.o: MMult5.c
	$(CC) $(CFLAGS) -fopenmp -c $< -o $@

MMult_openblas.o: MMult_openblas.c
	$(CC) $(CFLAGS) -I /opt/OpenBLAS/include/ -c $< -o $@

test_MMult.o: test_MMult.c parameters.h

time_MMult.o: time_MMult.c parameters.h

test_MMult_old.x: $(TEST) $(OLD).o $(UTILS)
	$(LINKER) $(TEST) $(OLD).o $(UTILS) $(LDFLAGS) -o $@

test_MMult_new.x: $(TEST) $(NEW).o $(UTILS)
	$(LINKER) $(TEST) $(NEW).o $(UTILS) $(LDFLAGS) -o $@

time_MMult_old.x: $(TIME) $(OLD).o $(UTILS)
	$(LINKER) $(TIME) $(OLD).o $(UTILS) $(LDFLAGS) -o $@

time_MMult_new.x: $(TIME) $(NEW).o $(UTILS)
	$(LINKER) $(TIME) $(NEW).o $(UTILS) $(LDFLAGS) -o $@

# export OMP_NUM_THREADS=16
# export GOTO_NUM_THREADS=1

# Just time the speed, do not validate correctness
run_time:
	make all

	echo "version = '$(OLD)';" > output_$(OLD).m
	echo $$OMP_NUM_THREADS
	./time_MMult_old.x >> output_$(OLD).m
	cp output_$(OLD).m output_old.m

	echo "version = '$(NEW)';" > output_$(NEW).m
	echo $$OMP_NUM_THREADS
	./time_MMult_new.x >> output_$(NEW).m
	cp output_$(NEW).m output_new.m

	python3 PlotAll.py

# Both time the speed and validate correctness
run_test:
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
