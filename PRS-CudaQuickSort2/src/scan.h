
/*
 * scan.h
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
 * this software uses the cutil library of NVIDIA CUDA SDK
 *
 * This file contains source code provided by NVIDIA Corporation
 * license: http://developer.download.nvidia.com/licenses/general_license.txt
 *
 *
 * the following Functions are derived by NVIDIA Corporation:
 *
 * 	 1. scanInclusiveLarge()
 * 	 2. scanInclusiveShort()
 * 	 3. warpScanInclusive2()
 * 	 4. warpScanExclusive2()
 * 	 5. scan1Inclusive2()
 * 	 6. warpCompareInclusive()
 * 	 7. compareInclusive
 *
 *
 *
 */

typedef unsigned int uint;

extern "C" size_t scanInclusiveShort(
    uint *d_Dst,
    uint *d_Src,
    uint batchSize,
    uint arrayLength
);

extern "C" size_t scanInclusiveLarge(
    uint *d_Dst,
    uint *d_Src,
    uint batchSize,
    uint arrayLength
);


template <typename Type>
inline __device__ void warpScanInclusive2(Type& idata,Type& idata2, volatile Type *s_Data,volatile Type *s_Data2, uint size){

    //volatile uint* s_Data2;
    //s_Data2 = s_Data + blockDim.x*2;

	uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    s_Data2[pos] = 0;
    pos += size;
    s_Data[pos] = idata;
    s_Data2[pos] = idata2;

    for(uint offset = 1; offset < size; offset <<= 1)
    {
    	s_Data[pos] += s_Data[pos - offset];
    	s_Data2[pos] += s_Data2[pos - offset];
    }

    idata=s_Data[pos];
    idata2=s_Data2[pos];
}

template <typename Type>
inline __device__ void warpScanExclusive2(Type& idata,Type& idata2, volatile Type *s_Data,volatile Type *s_Data2, uint size){

    //volatile uint* s_Data2;
    //s_Data2 = s_Data + blockDim.x*2;

	uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    s_Data2[pos] = 0;
    pos += size;
    s_Data[pos] = idata;
    s_Data2[pos] = idata2;

    for(uint offset = 1; offset < size; offset <<= 1)
    {
    	s_Data[pos] += s_Data[pos - offset];
    	s_Data2[pos] += s_Data2[pos - offset];
    }

    idata=s_Data[pos]-idata;
    idata2=s_Data2[pos]-idata2;
}

#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

template <typename Type>
inline __device__ void scan1Inclusive2(Type& idata,Type& idata2, volatile Type *s_Data, uint size){

    volatile Type* s_Data2;
    s_Data2 = s_Data + blockDim.x*2;

	if(size > WARP_SIZE){

    	//Bottom-level inclusive warp scan
        warpScanInclusive2(idata,idata2, s_Data,s_Data2, WARP_SIZE);

        //Save top Types of each warp for exclusive warp scan
        //sync to wait for warp scans to complete (because s_Data is being overwritten)
        __syncthreads();
        if( (threadIdx.x & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
        {
        	s_Data[threadIdx.x >> LOG2_WARP_SIZE] = idata;
        	s_Data2[threadIdx.x >> LOG2_WARP_SIZE] = idata2;
        }

        //wait for warp scans to complete
        __syncthreads();
        if( threadIdx.x < (blockDim.x / WARP_SIZE) ){
            //grab top warp Types
            Type val = s_Data[threadIdx.x];
            Type val2 = s_Data2[threadIdx.x];
            //calculate exclsive scan and write back to shared memory
            warpScanExclusive2(val,val2, s_Data,s_Data2, size >> LOG2_WARP_SIZE);
            s_Data[threadIdx.x] = val;
            s_Data2[threadIdx.x] = val2;
        }

        //return updated warp scans with exclusive scan results
        __syncthreads();
        idata  += s_Data[threadIdx.x >> LOG2_WARP_SIZE];
        idata2 += s_Data2[threadIdx.x >> LOG2_WARP_SIZE];
    }
    else
        warpScanInclusive2(idata,idata2, s_Data,s_Data2, size);

}

template <typename Type>
inline __device__ void warpCompareInclusive(Type& idata,Type& idata2, volatile Type *s_Data, uint size){

    volatile Type* s_Data2;
    s_Data2 = s_Data + blockDim.x*2;
	uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    s_Data2[pos] = 0;
    pos += size;
    s_Data[pos] = idata;
    s_Data2[pos] = idata2;

    for(uint offset = 1; offset < size; offset <<= 1)
    {
    	s_Data[pos] =max(s_Data[pos], s_Data[pos - offset]);
    	s_Data2[pos] =min(s_Data2[pos], s_Data2[pos - offset]);
    }

    idata = s_Data[pos];
    idata2= s_Data2[pos];
}

template <typename Type>
inline __device__ void compareInclusive(Type& idata,Type& idata2, volatile Type *s_Data, uint size){

    	volatile Type* s_Data2;
    	s_Data2 = s_Data + blockDim.x*2;
        //Bottom-level inclusive warp scan
        warpCompareInclusive(idata,idata2, s_Data, WARP_SIZE);

        //Save top Types of each warp for exclusive warp scan
        //sync to wait for warp scans to complete (because s_Data is being overwritten)
        __syncthreads();
        if( (threadIdx.x & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
         {
        	s_Data[threadIdx.x >> LOG2_WARP_SIZE] = idata;
        	s_Data2[threadIdx.x >> LOG2_WARP_SIZE] = idata2;
         }

        //wait for warp scans to complete
        __syncthreads();
        if( threadIdx.x < (blockDim.x / WARP_SIZE) ){
            //grab top warp Types
        	Type val = s_Data[threadIdx.x];
        	Type val2 = s_Data2[threadIdx.x];
            //calculate exclsive scan and write back to shared memory
             warpCompareInclusive(val,val2, s_Data, size >> LOG2_WARP_SIZE);
             s_Data[threadIdx.x] =val;
             s_Data2[threadIdx.x] =val2;
        }

        //return updated warp scans with exclusive scan results
        __syncthreads();
        idata=max(idata,s_Data[threadIdx.x >> LOG2_WARP_SIZE]) ;
        idata2=min(idata2,s_Data2[threadIdx.x >> LOG2_WARP_SIZE]) ;

}
