/*
 * randomDistr.h
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


#ifndef RANDOMDISTR_H_
#define RANDOMDISTR_H_

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <cmath>



using namespace std;

template <typename Type> inline
void bucket(Type* data, int N)
{

	srand (time(NULL));

	double tmp = ((double)N)*3000000; //(RAND_MAX)/p; --> ((double)N)*30000;
	double size = sqrt(tmp);

	//bucket number
	//std::fstream   fp;
	//fp.open("distr.csv", ios::out);
	unsigned int p= (N+size-1)/size;

	if(p==0)
	{
		std::cerr<<"Error: the bucket number is zero"<<std::endl;
		exit(1);
	}

	const unsigned int VALUE = 8192/p; //(RAND_MAX)/p;

	unsigned int i=0; int x=0;
	//the array of size N is split into 'p' buckets
	while(i < p)
	{
		for (unsigned int z = 0; z < p; ++z)
			for (unsigned int j = 0; j < N/(p*p); ++j)
			{
				//every bucket has N/(p*p) items and the range is [min : VALUE-1 ]
				unsigned int min = VALUE*z;

				data[x]= min + ( rand() %  (VALUE-1) ) ;
				//fp<<data[x]<<endl;
				x++;
			}
		i++;
	}
	//fp.close();

}

template <typename Type>
void staggared(Type* data, int N)
{
	srand (time(NULL));
	//numero di bucket
	int size=4096; //(RAND_MAX)/p; --> size=2048
	unsigned int p= (N+size-1)/size;

	if(p==0)
	{
		std::cerr<<"Error: the bucket number is zero"<<std::endl;
		exit(1);
	}

	const unsigned int VALUE = 16384/p; //(RAND_MAX)/p;

	unsigned int i=1; int x=0;
	//the array of size N is split into 'p' buckets
	while(i <= p)
	{
		//every bucket has N/(p) items
		for (unsigned int j = 0; j < N/(p); ++j)
		{
			unsigned int min;

			if(i<=(p/2))
				min = (2*i -1)*VALUE;

			else
				min = (2*i-p-1)*VALUE;

			data[x++]= min + ( rand() % (VALUE - 1) );

		}
		i++;
	}


}

template <typename Type> inline
void zero (Type* data,int N)
{
	srand (time(NULL));
	Type value = rand();
	for (int i = 0; i < N; ++i)
		data[i] = value ;
}

template <typename Type> inline
void uniform(Type* data,int N)
{
	srand (time(NULL));
	for (int i = 0; i < N; ++i)
		data[i] = (Type)(rand()%16384  )  ;
}

template <typename Type> inline
void sorted(Type* data,int N)
{
	srand (time(NULL));
	for (int i = 0; i < N; ++i)
		data[i] = rand()%16384 ;

	std::sort(data,data+N);
}

template <typename Type> inline
void inverseSorted(Type* data, int N)
{
	srand(time(NULL));
	for (int i = 0; i < N; ++i)
		data[i] = rand() % 16384;

	std::sort(data, data + N, std::greater<>());
}

template <typename Type> inline
void gaussian(Type* data,int N)
{
	srand (time(NULL));
	unsigned int value = 0;
	for (int i = 0; i < N; ++i) {
		value = 0;

		for (int j = 0; j < 4; ++j) {
			value += rand()%16384;
		}

		data[i] = value /4;

	}

}

template <typename Type>
void distribution(Type* data, int dataSize, std::string& typeDist)
{
	if(typeDist == "uniform")
	{
		uniform<Type>(data,dataSize);
		return;
	}

	if(typeDist == "bucket" )
	{
		bucket(data, dataSize);
		return;
	}

	if(typeDist == "staggered")
	{
		staggared(data,dataSize);
		return;
	}

	if(typeDist == "sorted")
	{
		sorted(data,dataSize);
		return;
	}

	if (typeDist == "inverseSorted") {
		inverseSorted(data, dataSize);
		return;
	}

	if(typeDist == "zero")
	{
		zero(data,dataSize);
		return;
	}

	if(typeDist == "gaussian")
	{
		gaussian(data,dataSize);
		return;
	}

	std::cout<<"distribution don't match, will be used the uniform distribution"<<std::endl;

	uniform(data, dataSize);

	return;

}




#endif /* RANDOMDISTR_H_ */
