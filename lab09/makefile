OLD  := MMult5
NEW  := MMult6

CC         := gcc
LINKER     := $(CC)
CFLAGS     := -Wall -g -O1
LDFLAGS    := -lm -L /opt/OpenBLAS/lib/ -lopenblas -fopenmp

TEST       := test_MMult.o
TEST_OBJ   := MMult5.o MMult6.o MMult_openblas.o
UTILS      := utils.o
EXECUTABLE := test_MMult_old.x test_MMult_new.x


all:
	make $(UTILS) $(TEST_OBJ) $(TEST)
	rm -f *.x
	make $(EXECUTABLE)

test_MMult.o: test_MMult.c parameters.h dgemm.h

MMult5.o: MMult5.c dgemm.h
	$(CC) $(CFLAGS) -fopenmp -c $< -o $@

MMult6.o: MMult6.c dgemm.h
	$(CC) $(CFLAGS) -fopenmp -c $< -o $@

MMult_openblas.o: MMult_openblas.c dgemm.h
	$(CC) $(CFLAGS) -I /opt/OpenBLAS/include/ -c $< -o $@

utils.o: utils.c utils.h dgemm.h MMult_openblas.o

test_MMult_old.x: $(TEST) $(OLD).o $(UTILS) parameters.h dgemm.h utils.h
	$(LINKER) $(TEST) $(OLD).o $(UTILS) $(LDFLAGS) -o $@

test_MMult_new.x: $(TEST) $(NEW).o $(UTILS) parameters.h dgemm.h utils.h
	$(LINKER) $(TEST) $(NEW).o $(UTILS) $(LDFLAGS) -o $@

test time:
	make all

	echo "version = '$(OLD)';" > output_$(OLD).m
	echo $$OMP_NUM_THREADS
	./test_MMult_old.x --$@ >> output_$(OLD).m
	cp output_$(OLD).m output_old.m

	echo "version = '$(NEW)';" > output_$(NEW).m
	echo $$OMP_NUM_THREADS
	./test_MMult_new.x --$@ >> output_$(NEW).m
	cp output_$(NEW).m output_new.m

	python3 PlotAll.py

clean:
	rm -f *.o *~ core *.x

cleanall:
	rm -f *.o *~ core *.x output*.m *.eps *.png *.jpg
