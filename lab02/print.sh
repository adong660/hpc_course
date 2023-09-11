if [ -f ./timeDGEMM.txt ]
then
	rm ./timeDGEMM.txt
fi

for size in 16 64 256 1024 2048
do
	./time_dgemm ${size}
done

exit 0

