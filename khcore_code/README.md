# khcore

Note: Cloned from https://github.com/BITDataScience/khcore.

This code can be compiled with g++. If the compilation is not successful, please use the later version of g++ and c++11 or higher standard libraries.

Compilation: 

	g++ "-std=c++11" -O3 -o run KHCore.cpp -fopenmp

The first parameter is the algorithm that the program runs. Please select one of {txt-to-bin, decompose, compare, scal}

	txt-to-bin: format the txt file to the bin file.
	decompose: run the decomposition algorithms.
	compare: compute the precision of the sampling algrithm.
	scal: the scalability testing.

Running examples: 

	./run txt-to-bin data/example.txt
	./run decompose data/example.bin 2
	./run decompose data/example.bin 2 4 0.1 0.01
	./run decompose data/example.bin 2 4 0.1 0.01 data/ core-number.bin
	./run compare path file1 file2

If running 'scal'

	usage: ./run scal scal-rate vary infile h t sampling-rate error-rate outfile-path outfile
	scal-rate: the range in (0,100].
	vary: '1' for sampling vertices, or '2' for sampling edges.
