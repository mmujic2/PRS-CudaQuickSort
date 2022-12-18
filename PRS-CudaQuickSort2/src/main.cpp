/*
 * main.cpp
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
 * 		3. thrust::sort()
 *
 *
 *
 * this software uses the library of NVIDIA CUDA SDK and the Cederman and Tsigas' GPU Quick Sort
 *
 */
# include <cstdlib>

#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>

#include <gpuqsort.h>
#include "randomDistr.h"
#include "CUDA-Quicksort.h"


using namespace std;


bool validateSortedValue(Type* dataTest, Type* data, unsigned int dataSize)
{

	unsigned int i = 0;

	while(i<dataSize)
	{
		if( data[i] != dataTest[i] )
		{//cout<<i<<' '<<data[i]<<endl;
			cout<<"Error: dataTest != data"<<endl;
			//cerr<<"dataTest["<<i<<"] ==\t"<< dataTest[i]<<endl;
			//cerr<<"data    ["<<i<<"} ==\t"<< data[i]    <<endl;
			//getchar();
			return false;
		}
		i++;
	}

	return true;
}



//Testing on all distributions for NUMBER_DATASET times
void compareTest(string& select,int device)
{
	//max 24
	const unsigned int N = 2<<24;
	const int NUMBER_DATASET=1;

	fstream   fp;
	fp.open("compareTest.csv", ios::out);

	if(!fp)
	{
	    cerr << "file don't open"<<endl;
	    fp.close();
	    return;
	 }


	Type* inData   = new Type[N];
	Type* outData  = new Type[N];

	double timerQuick,timerQuick2,timerBitonic,timerMerge;
	//float timerCdpQuicksort;


	string distr[6]={"uniform","gaussian","zero","bucket","staggered","sorted"};

	cout.setf(ios_base::fixed, ios_base::floatfield);
	fp.setf  (ios_base::fixed, ios_base::floatfield);
	cout.precision(3);
	fp.precision(3);

	cout<<"running Testing on all distributions ...\n\n";

	float count=0;
	unsigned int i_dist = 0;

	unsigned int q   = (select == "all") ? 6 : 1;


	while( i_dist < 6 )
	{
		if( distr[i_dist] != select && select!= "all") { i_dist++; continue;}

		fp<<"\n'distribution "<<distr[i_dist]<<", average time on "<<NUMBER_DATASET<<"\n"<<endl;
		//fp<<"dataSize\tRadixSort\tCUDA-Quicksort\tGPU-QuickSort\tCDP-Quicksort\tBitonicSort\tMergeSort"<<endl;
		fp<<"dataSize\tRadixSort\tCUDA-Quicksort\tGPU-QuickSort\tBitonicSort\tMergeSort"<<endl;

		unsigned int dataSize = 2<<19;
		while(dataSize<=N)
		{

			double totBitonic=0;
			double totMerge=0;
			double totQuick=0;
			double totQuick2=0;
			//float totcdpQuick=0;


			cout<<"\n"<<distr[i_dist]<<" distribution\n"<<"dataSize: "<<dataSize;

			for(int i=0;i<NUMBER_DATASET;i++)
			{

				distribution<Type>(inData,dataSize,distr[i_dist]);

				if(dataSize<(2<<22))
					test_MergeSort(inData,dataSize,&timerMerge);
				else timerMerge=0;

				if(dataSize<(2<<24))
					test_bitonicSort(inData,dataSize,&timerBitonic);
				else timerBitonic=0;

				// threads  =128; threshold   =512;
				CUDA_Quicksort(inData,outData, dataSize,128,device,&timerQuick2);

				//test_cdpQuicksort(inData, dataSize,&timerCdpQuicksort);

				gpuqsort(inData,dataSize,&timerQuick,0,0,0,0);

    			totBitonic+=timerBitonic;
				totMerge+=timerMerge;
				totQuick+=timerQuick;
				totQuick2+=timerQuick2;
				//totcdpQuick+=timerCdpQuicksort;

				count++;
			}

			fp<<dataSize;

			fp<<"\t"<< totQuick2  /NUMBER_DATASET;
			fp<<"\t"<< totQuick   /NUMBER_DATASET;
			//fp<<"\t"<< totcdpQuick  /NUMBER_DATASET;
			fp<<"\t"<< totBitonic /NUMBER_DATASET;
			fp<<"\t"<< totMerge   /NUMBER_DATASET<<endl;



			cout<<"INFO: Average time of the CUDA-Quicksort "<<totQuick2/NUMBER_DATASET<<" ms"<<endl;
			//cout<<"Complete "<<(count/(15*q*NUMBER_DATASET))*100<<"%"<<endl;
			dataSize*=2;
		}
		i_dist++;
	}
	fp.close();
	delete inData;
	delete outData;
}




void validationTest(unsigned int size, string& select,int device)
{
	const unsigned int N = size ==0 ? 2<<24 : size;

	const int NUMBER_DATASET=1;

	Type* inData  = new Type[N];
	Type* outData = new Type[N];


	double timerQuick;
	bool validate=false;;

	string distr[6]={"uniform","gaussian","zero","bucket","staggered","sorted"};

	cout.setf(ios_base::fixed, ios_base::floatfield);
	cout.precision(3);

	cout<<"running validation test ...\n\n";


	float count=0;

	unsigned int i_dist   = 0;
	unsigned int q   = (select == "all") ? 6 : 1;
	//unsigned int dataSize = (size == 0) ? 2<<10 : size;

	while( i_dist < 6 )
	{
		if( distr[i_dist] != select && select!= "all") { i_dist++; continue;}

		cout<<"\nvalidation the results on "<<NUMBER_DATASET<<" dataset for the "<<distr[i_dist]<< " distribution ...\n"<<endl;

		unsigned int dataSize = (size == 0) ? 2<<19 : size;
		while(dataSize<=N)
		{

			double totQuick=0;

			cout<<"\n"<<distr[i_dist]<< " distribution  dataSize: "<<dataSize<<endl;

			for(int i=0;i<NUMBER_DATASET;i++)
			{

				distribution<Type>(inData,dataSize,distr[i_dist]);

				// threads  =128;
				CUDA_Quicksort(inData,outData, dataSize,128,device,&timerQuick);

				sort(inData,inData+dataSize);
				validate = validateSortedValue(inData,outData,dataSize);

				if(!validate)
				{
					cout<<"Test Fault on dataset "<<i<<endl<<endl;
					totQuick =0;
					count++;
					continue;
				}

				totQuick+=timerQuick;
				count++;
			}

			if(validate)
				cout<<"test Pass...\t"<<"average time "<<totQuick/NUMBER_DATASET<<" ms"<<endl;

			dataSize*=2;
		}
		i_dist++;
	}
	delete inData;
	delete outData;
}

void validationTest_64(unsigned int size, string& select,int device)
{
	const unsigned int N = size ==0 ? 2<<24 : size;

	const int NUMBER_DATASET=1;

	double* inData  = new double[N];
	double* outData = new double[N];


	double timerQuick;
	bool validate=false;;

	string distr[6]={"uniform","gaussian","zero","bucket","staggered","sorted"};

	cout.setf(ios_base::fixed, ios_base::floatfield);
	cout.precision(3);

	cout<<"running validation test...\n\n";


	float count=0;

	unsigned int i_dist   = 0;
	unsigned int q   = (select == "all") ? 6 : 1;
	//unsigned int dataSize = (size == 0) ? 2<<10 : size;

	while( i_dist < 6 )
	{
		if( distr[i_dist] != select && select!= "all") { i_dist++; continue;}

		cout<<"\nvalidation the results on "<<NUMBER_DATASET<<" dataset for the "<<distr[i_dist]<< " distribution ...\n"<<endl;

		unsigned int dataSize = (size == 0) ? 2<<10 : size;
		while(dataSize<=N)
		{

			double totQuick=0;

			cout<<"\n"<<distr[i_dist]<< " distribution  dataSize: "<<dataSize<<endl;

			for(int i=0;i<NUMBER_DATASET;i++)
			{

				distribution<double>(inData,dataSize,distr[i_dist]);

				// threads  =256;
				CUDA_Quicksort_64(inData,outData, dataSize,256,device,&timerQuick);

				sort(inData,inData+dataSize);
				validate = true;//validateSortedValue(inData,outData,dataSize);

				if(!validate)
				{
					cout<<"Test Fault on dataset "<<i<<endl<<endl;
					totQuick =0;
					count++;
					continue;
				}

				totQuick+=timerQuick;
				count++;
			}

			if(validate)
				cout<<"test Pass...\t"<<"average time "<<totQuick/NUMBER_DATASET<<" ms"<<endl;
			//cout<<"Complete "<<(count/(15*q*NUMBER_DATASET))*100<<"%"<<endl;
			dataSize*=2;
		}
		i_dist++;
	}
	delete inData;
	delete outData;
}



void GPUtest(unsigned int size,string& select,int device)
{
	const unsigned int N = size ==0 ? 2<<24 : size;

	Type* inData    =  new Type[N];
	Type* outData   =  new Type[N];
	Type* datatest  =  new Type[N];

	double timerQuick;
	string distr;

	if(select == "all")
		distr = "uniform";
	else
		distr = select;


	distribution(inData,N,distr);

	memcpy(datatest,inData,N*sizeof(unsigned int));


	unsigned int dataSize = size ==0 ? 2<<19 : size;
	while( dataSize<=N )
	{
		cout<<"\ndataSize: "<<dataSize<<"\tdistribution: "<<distr<<endl;
		//CUDA-QuickSort works only for thread=128|256 if the shared memory size is 1024 (see SHARED_LIMIT on CUDA-Quicksort.h).
		//The limit is thread*4<=SHARED_LIMIT
		for(int threads=128; threads<=256 ; threads*=2)
				{
					CUDA_Quicksort(inData,outData,dataSize,threads,device,&timerQuick);

					cout<<"time: "<<timerQuick<<" ms";
					cout<<"\tthreads: "<<threads<<endl;

					sort(datatest,datatest+dataSize);
					validateSortedValue(datatest,outData,dataSize);
				}
		dataSize *= 2;

	}

	delete inData;
	delete outData;
	delete datatest;
}

void help()
{
	  int length;
	  char * buffer;

	  ifstream is;
	  is.open ("../README.txt", ios::binary );

	  if(!is)
	  {
		  is.open ("README.txt", ios::binary );
		  if(!is)
		  {
			  is.close();
			  return;
		  }
	  }
	  // get length of file:
	  is.seekg (1321, ios::end);
	  length = is.tellg();
	  is.seekg (1321, ios::beg);

	  // allocate memory:
	  buffer = new char [length];
	  // read data as a block:
	  is.read (buffer,length);
	  is.close();

	  cout.write (buffer,length);

	  delete[] buffer;


}


int main(int argc,const char* argv[])
{

	// -- Initialise -- //
	/*string distribution = "all";
	unsigned int size = 0;
	char flag = 't';
	int device=0;

	for (int argList = 1; argList < argc; ++argList)
	{
	    if (argv[argList][0] == '-' && argv[argList][1] != '-')
	    {
	         for (int j = 1; argv[argList][j] != '\0'; j++)
	         {
				  switch (argv[argList][j])
				  {
				  	  case 'a': distribution = "all";		break;
					  case 'u': distribution = "uniform";	break;
					  case 'g': distribution = "gaussian";	break;
					  case 'z': distribution = "zero";		break;
					  case 'r': distribution = "staggered"; break;
					  case 'b': distribution = "bucket"; 	break;
					  case 's': distribution = "sorted";	break;

	      	  	  	  case 't': flag = argv[argList][1]; 	break;
	      	  	  	  case 'v': flag = argv[argList][1]; 	break;
	      	  	  	  case 'c': flag = argv[argList][1]; 	break;

	      	  	  	  case '1': size = 2<<19; break;
			      	  case '2': size = 2<<20; break;
			      	  case '3': size = 2<<21; break;
			      	  case '4': size = 2<<22; break;
			      	  case '5': size = 2<<23; break;
			      	  case '6': size = 2<<24; break;

			          default: cerr<<"Bad flag: "<<argv[argList]<<endl<<endl<<endl;
			        		   help();
			        		   return 0;
				  }
	         }
	    }
	    else
	    if (argv[argList][0] == '-' && argv[argList][1] == '-')
	    {
	    	if( string(argv[argList]) == "--all" )  	 distribution = "all";			else
	    	if( string(argv[argList]) == "--uniform" )   distribution = "uniform";		else
	    	if( string(argv[argList]) == "--gaussian" )  distribution = "gaussian"; 	else
	    	if( string(argv[argList]) == "--zero" )      distribution = "zero"; 	   	else
	    	if( string(argv[argList]) == "--staggered" ) distribution = "staggered";	else
	    	if( string(argv[argList]) == "--bucket" )    distribution = "bucket";		else
	    	if( string(argv[argList]) == "--sorted" )    distribution = "sorted";		else

	    	if( string(argv[argList]) == "--compare" )   flag = 'c';					else
	    	if( string(argv[argList]) == "--testGPU" )   flag = 't';					else
	    	if( string(argv[argList]) == "--validate" )  flag = 'v';					else

	    	if( string(argv[argList]) == "--1M" )        size=2<<19;					else
	    	if( string(argv[argList]) == "--2M" )        size=2<<20;					else
	    	if( string(argv[argList]) == "--4M" )        size=2<<21;					else
	    	if( string(argv[argList]) == "--8M" )        size=2<<22;					else
	    	if( string(argv[argList]) == "--16M" )       size=2<<23;					else
	    	if( string(argv[argList]) == "--32M" )       size=2<<24;					else

	    	if( string(argv[argList]) == "--device" )    device=atoi(argv[argList+1]);					else
	        if( string(argv[argList]) == "--size" )      size=atoi(argv[argList+1]);					else

	    	if( string(argv[argList]) == "--help" )      {help(); return 0;}			else

	    	{ cerr<<"Bad flag: "<<argv[argList]<<endl; return 0; }


	    }
	    //else istringstream ( argv[argList] ) >> size ;
	}


	if (flag == 't')
	{
		GPUtest(size,distribution,device);
		return 0;
	}


	if( flag == 'v' )
	{

		validationTest(size,distribution,device);
		return 0;
	}

	if( flag == 'c' )
	{
		compareTest(distribution,device);
		return 0;
	}*/

	string ds[1] = {"gaussian"};
	unsigned int sizes[1] = { (2 << 20) };

	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 1; j++) {
			GPUtest(sizes[j], ds[i], 0);
		}
	}
	// 2^25 = 64M 2 * 2 * 1.5
	// 2^26 = 128M
	// 2^27 = 256M * 1.5 = 400M
	// 2^28 = 512M



	return 0;
}



