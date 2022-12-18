/*
 * CUDA-Quicksort.h
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
 * this software uses the library of NVIDIA CUDA SDK and the Cederman and Tsigas' GPU Quick Sort
 *
 */



//defines the shared memory size
#define SHARED_LIMIT 1024

#define GIGA 1073741824
/*
 * division of the vector to be sorted in buckets
 * the attributes of the object Block are the parameters of each bucket
 */
template <typename Type>
struct Block
{

	unsigned int begin;
	unsigned int end;

	unsigned int nextbegin;
	unsigned int nextend;

	Type		 pivot;

	//max of the bucket items
	Type		 maxPiv;
	//min of the bucket items
	Type		 minPiv;
	//done indicates that a bucket has been analyzed
	short		 done;
	short		 select;


};



template <typename Type>
struct Partition
{

	unsigned int ibucket;
	unsigned int from;
	unsigned int end;
	Type pivot;
};



extern "C"
void CUDA_Quicksort(unsigned int* inData, unsigned int* outData, unsigned int dataSize, unsigned int threads, int Device, double* timer);

extern "C"
void CUDA_Quicksort_64(double* inData,double* outData, unsigned int size, unsigned int threads, int Device, double* timer);

typedef unsigned int Type;

extern "C" void test_bitonicSort(unsigned int* h_InputKey,unsigned int N, double* timer);
extern "C" void test_MergeSort  (unsigned int*h_SrcKey   ,unsigned int N, double* timer);
extern "C" void test_thrustSort (Type* h_data    ,unsigned int N, double* timer);

//extern "C" int  test_cdpQuicksort(unsigned int* data,unsigned int size,float* timer);

