################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/main.cpp \
../src/sortingNetworks_validate.cpp 

CU_SRCS += \
../src/bitonicSort.cu \
../src/oddEvenMergeSort.cu 

CU_DEPS += \
./src/bitonicSort.d \
./src/oddEvenMergeSort.d 

OBJS += \
./src/bitonicSort.o \
./src/main.o \
./src/oddEvenMergeSort.o \
./src/sortingNetworks_validate.o 

CPP_DEPS += \
./src/main.d \
./src/sortingNetworks_validate.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I../../include/ -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -I../../include/ -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I../../include/ -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I../../include/ -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


