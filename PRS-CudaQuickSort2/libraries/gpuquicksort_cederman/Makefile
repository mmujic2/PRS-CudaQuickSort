LIB_SOURCES 	= gpuqsort.cu
TEST_SOURCES 	= main.cpp dists.cpp
LIB_OBJ		= $(LIB_SOURCES:.cu=.o)
TEST_OBJ	= $(TEST_SOURCES:.cpp=.o)

all: gpuqsortlib testbench

gpuqsortlib: $(LIB_OBJ)
	mkdir -p lib/linux/static
	mkdir -p lib/linux/shared
	nvcc -o lib/linux/static/libgpuqsort.a -lib $(LIB_OBJ)
	nvcc -o lib/linux/shared/libgpuqsort.so -shared $(LIB_OBJ)

testbench: $(TEST_OBJ) gpuqsortlib
	mkdir -p bin/linux
	nvcc -o bin/linux/testbench $(TEST_OBJ) -L./lib/linux/static -lgpuqsort 
#-lglut 
# -lsqlite3

%.o: src/%.cu
	nvcc -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=sm_50 -I$(CUDA_SDK_DIR)/common/inc -Xcompiler -fPIC -Iinc -c $< -o $@

%.o: testbench/%.cpp
	gcc -O3 -Iinc -c $< -o $@

clean:

	rm -rf ./gpuqsort.o ./dists.o ./main.o 
	-@echo ' ' 
