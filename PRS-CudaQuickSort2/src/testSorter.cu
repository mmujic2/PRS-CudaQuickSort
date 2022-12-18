/*
 * testSorter.cu
 *
 * Copyright © 2012-2015 Emanuele Manca
 *
 **********************************************************************************************
 **********************************************************************************************
 *
 	This file is part of CUDA-Quicksort.

    CUDA-Quicksort is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDA-Quicksort is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CUDA-Quicksort.

    If not, see http://www.gnu.org/licenses/gpl-3.0.txt and http://www.gnu.org/copyleft/gpl.html


  **********************************************************************************************
  **********************************************************************************************
 *
 * Contact: Ing. Emanuele Manca
 *
 * Department of Electrical and Electronic Engineering,
 * University of Cagliari,
 * P.zza D’Armi, 09123, Cagliari, Italy
 *
 * email: emanuele.manca@diee.unica.it
 *
 *
 * This software contains source code provided by NVIDIA Corporation
 * license: http://developer.download.nvidia.com/licenses/general_license.txt
 *
 * the following Functions are based or derived from the NVIDIA CUDA SDK:
 *
 * 		1. bitonicSort()
 * 		2. mergesort()
 * 		3. thrust::Sort()
 *
 *
 * this software uses the library of NVIDIA CUDA SDK and the Cederman and Tsigas' GPU Quick Sort 
 *
 */


#include "CUDA-Quicksort.h"

#include <sortingNetworks_common.h>
#include <mergeSort_common.h>
#include <helper_cuda.h>
#include <helper_timer.h>
//#include <cdpQuicksort.h>





void test_bitonicSort(uint* h_InputKey,uint N,double* timer)
{

	StopWatchInterface* htimer=NULL;
	uint* h_InputVal =  (uint *)malloc(N * sizeof(uint));
	//uint* h_test     =  (uint *)malloc(N * sizeof(uint));

	for(uint i = 0; i < N; i++)
		h_InputVal[i] = i;


	uint*d_InputKey,*d_InputVal,*d_OutputKey,*d_OutputVal;
	sdkCreateTimer(&htimer);
	checkCudaErrors( cudaMalloc((void **)&d_InputKey,  N * sizeof(uint)) );
	checkCudaErrors( cudaMalloc((void **)&d_InputVal,  N * sizeof(uint)) );
	checkCudaErrors( cudaMalloc((void **)&d_OutputKey, N * sizeof(uint)) );
	checkCudaErrors( cudaMalloc((void **)&d_OutputVal, N * sizeof(uint)) );

	checkCudaErrors( cudaMemcpy(d_InputKey, h_InputKey, N * sizeof(uint), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_InputVal, h_InputVal, N * sizeof(uint), cudaMemcpyHostToDevice) );

	checkCudaErrors(cudaDeviceSynchronize());
	sdkCreateTimer(&htimer);
    sdkResetTimer(&htimer);
    sdkStartTimer(&htimer);


	//printf("Initializing GPU Bitonic sort...\n");
	bitonicSort(d_OutputKey,
                d_OutputVal,
                d_InputKey,
                d_InputVal,
                N / N,
                N,
                1
                );

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&htimer);
    *timer=sdkGetTimerValue(&htimer);

	//checkCudaErrors( cudaMemcpy(h_test, d_OutputKey, N * sizeof(uint), cudaMemcpyDeviceToHost) );

	checkCudaErrors( cudaFree(d_OutputVal) );
	checkCudaErrors(    cudaFree(d_OutputKey) );
	checkCudaErrors(    cudaFree(d_InputVal) );
	checkCudaErrors(    cudaFree(d_InputKey) );
    //free(h_test);
    free(h_InputVal);

}

void test_MergeSort(uint*h_SrcKey,uint N,double* timer)
{

	uint  *h_SrcVal, *h_DstKey, *h_DstVal;
	uint *d_SrcKey, *d_SrcVal, *d_BufKey, *d_BufVal, *d_DstKey, *d_DstVal;


    h_SrcVal = (uint *)malloc(N * sizeof(uint));
    h_DstKey = (uint *)malloc(N * sizeof(uint));
    h_DstVal = (uint *)malloc(N * sizeof(uint));


    for(uint i = 0; i < N; i++)
             h_SrcVal[i] =i;

   const uint DIR = 1;
   StopWatchInterface* htimer=NULL;

   checkCudaErrors( cudaMalloc((void **)&d_DstKey, N * sizeof(uint))) ;
   checkCudaErrors( cudaMalloc((void **)&d_DstVal, N * sizeof(uint)) );
   checkCudaErrors( cudaMalloc((void **)&d_BufKey, N * sizeof(uint)) );
   checkCudaErrors( cudaMalloc((void **)&d_BufVal, N * sizeof(uint)) );
   checkCudaErrors( cudaMalloc((void **)&d_SrcKey, N * sizeof(uint)) );
   checkCudaErrors( cudaMalloc((void **)&d_SrcVal, N * sizeof(uint)) );

   checkCudaErrors(cudaMemcpy(d_SrcKey, h_SrcKey, N * sizeof(uint), cudaMemcpyHostToDevice) );
   checkCudaErrors(cudaMemcpy(d_SrcVal, h_SrcVal, N * sizeof(uint), cudaMemcpyHostToDevice) );

    //Initializing GPU merge sort...
    initMergeSort();

	checkCudaErrors(cudaDeviceSynchronize());
	sdkCreateTimer(&htimer);
    sdkResetTimer(&htimer);
    sdkStartTimer(&htimer);

    mergeSort(d_DstKey, d_DstVal,d_BufKey,d_BufVal,d_SrcKey,d_SrcVal, N,DIR);

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&htimer);
    *timer=sdkGetTimerValue(&htimer);

    checkCudaErrors( cudaMemcpy(h_DstKey, d_DstKey, N * sizeof(uint), cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaMemcpy(h_DstVal, d_DstVal, N * sizeof(uint), cudaMemcpyDeviceToHost) );


    closeMergeSort();

    checkCudaErrors( cudaFree(d_SrcVal) );
    checkCudaErrors( cudaFree(d_SrcKey) );
    checkCudaErrors( cudaFree(d_BufVal) );
    checkCudaErrors( cudaFree(d_BufKey) );
    checkCudaErrors( cudaFree(d_DstVal) );
    checkCudaErrors( cudaFree(d_DstKey) );
    free(h_DstVal);
    free(h_DstKey);
    free(h_SrcVal);

}




/*void test_thrustSort(Type* data, uint N, double* timer)
{

	//Type* h_data=(Type *)malloc(N * sizeof(Type));
	Type* d_data;

	checkCudaErrors( cudaMalloc((void**)&d_data,(N)*sizeof(Type)) );
	checkCudaErrors( cudaMemcpy(d_data, data, N*sizeof(Type), cudaMemcpyHostToDevice) ) ;

	StopWatchInterface* htimer=NULL;

	checkCudaErrors(cudaDeviceSynchronize());
	sdkCreateTimer(&htimer);
    sdkResetTimer(&htimer);
    sdkStartTimer(&htimer);


	thrust::device_ptr<Type> array(d_data);
	thrust::sort(array,array + N);
	getLastCudaError("thrust::sort() execution FAILED\n");


	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&htimer);
    *timer=sdkGetTimerValue(&htimer);

	//checkCudaErrors( cudaMemcpy(h_data, d_data, N*sizeof(Type), cudaMemcpyDeviceToHost) );

	//thrust::device_free(array);
	checkCudaErrors( cudaFree(d_data));
	//free(h_data);

}*/


/*

int test_cdpQuicksort(unsigned int* data,unsigned int size,float* timer)
{


    // Create and set up our test
    unsigned *gpudata, *scratchdata;
    checkCudaErrors(cudaMalloc((void **)&gpudata, size*sizeof(unsigned)));
    checkCudaErrors(cudaMalloc((void **)&scratchdata, size*sizeof(unsigned)));

    checkCudaErrors(cudaMemcpy(gpudata, data, size*sizeof(unsigned), cudaMemcpyHostToDevice));

    (*timer) = run_quicksort_cdp(gpudata, scratchdata, size, NULL);


    checkCudaErrors(cudaDeviceSynchronize());

    // Copy back the data and verify correct sort
    //checkCudaErrors(cudaMemcpy(data, gpudata, size*sizeof(unsigned), cudaMemcpyDeviceToHost));


    // Release everything and we're done
    checkCudaErrors(cudaFree(scratchdata));
    checkCudaErrors(cudaFree(gpudata));

    return 0;
}
*/
