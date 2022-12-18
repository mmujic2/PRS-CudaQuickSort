################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/main.cpp \
../src/mergeSort_host.cpp \
../src/mergeSort_validate.cpp 

CU_SRCS += \
../src/bitonic.cu \
../src/mergeSort.cu 

CU_DEPS += \
./src/bitonic.d \
./src/mergeSort.d 

OBJS += \
./src/bitonic.o \
./src/main.o \
./src/mergeSort.o \
./src/mergeSort_host.o \
./src/mergeSort_validate.o 

CPP_DEPS += \
./src/main.d \
./src/mergeSort_host.d \
./src/mergeSort_validate.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I../../include/ -O3 -gencode arch=compute_20,code=sm_20 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -I../../include/ -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I../../include/ -O3 -gencode arch=compute_20,code=sm_20 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I../../include/ -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


