CU_SRC = genBuckets.cu texMark.cu
CU_OBJS = $(CU_SRC:.cu=.o)

%.o: %.cu
	nvcc -arch=sm_35 -c $< -o $@

default: texMark 

texMark: $(CU_OBJS)
	nvcc -arch=sm_35 -lboost_regex $^ -o tmBin

clean:
	rm -f tmBin *.o 
