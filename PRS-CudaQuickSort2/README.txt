				** CUDA-Quicksort **
			*       version 1.6.1        *


--------------------------------------------------------------------------------
SYSTEM REQUIREMENTS
--------------------------------------------------------------------------------

1. OS: linux, Windows

2. CPU: all

3. GPU: NVIDIA Fermi GPU architecture or higher (2.x capability or higher)

4. Builder: CUDA 5.5 or higher. For Older CUDA versions we recommend using Nsight Eclipse Edition (for linux) or Nsight Visual Studio Edition (for Windows)

NOTE: It is recommended to use NVIDIA Kepler GPU architecture with CUDA 5.5; using CUDA 4.0 or 4.2 would result in a slower runtime

--------------------------------------------------------------------------------
HOW TO BUILD : LINUX
--------------------------------------------------------------------------------

1. You can build the project through Nsight Eclipse Edition

2. The makefile can be found in the Release_linux folder. 
   NOTE: This makefile works only with CUDA 6.0 or higher. You can use CUDA 5.5 but you must delete "-gencode arch=compute_32,code=sm_32" and "-gencode arch=compute_50,code=sm_50" in the file "Release_linux/src/subdir.mk" . Older CUDA versions do not support this makefile

3. Type 'make'.  This builds the CUDA-Quicksort:

	make all   - build all projects (the executable file will be created in the Relese_linux folder)
	make clean - clean project

	
--------------------------------------------------------------------------------
HOW TO BUILD : Windows
--------------------------------------------------------------------------------

1. You can build the project through Microsoft Visual Studio and Nsight Visual Studio Edition 

2. The executable file will be created in the Relese_win\Win32 folder

--------------------------------------------------------------------------------
HOW TO RUN : LINUX or Windows
--------------------------------------------------------------------------------
	

To run the tests on CUDA-Quicksort, type the following: ./cuda-quickSort [test Flags] [distribution flags] [size flags] [--size SIZE] [--device identifier]


[test flags]:

	default (no flag) or -t or --testGPU 

	 	(run testGPU) The CUDA-Quicksort performance is compared over several configurations, e.g.:
			1. number of threads 
			2. size of the array to be sorted

	-c or --compare:
	
		(run compare test) The CUDA-Quicksort performance is compared to the Radix Sort, Merge Sort, Bitonic Sort and Cederman and Tsigas' GPU Quick Sort performance.
		Results are stored in the following file: [executable folder]/compareTest.txt
	
	-v or --validate
	
		 (run validate test) A validity check is performed on CUDA-Quicksort


[distribution flags]:

	default or -a or --all
	
		The CUDA-Quicksort performance is compared over the following distributions: [uniform, gaussian, zero, staggered, bucket, sorted]
		
	-u or --uniform
	
		The input will be a uniformly distributed random vector
		
	-g or --gaussian
	
		The input will be a Gaussian distributed random vector

	-s or --sorted
	
		The input will be a sorted vector	
		
	-b or --bucket
	
		The input will be a vector sorted into p buckets
		
	-z or --zero
	
		The input will be a zero entropy vector
		
	-r or --staggered
	
		The input will be a Staggered distributed random vector

		

[size flags]:
(size flags work only with testGPU and validate test, in compare test the size flags are "skipped")

	default (no flags) 
	
		Tests are performed on vectors of the following sizes: [1M, 2M, 4M, 8M, 16M, 32M]
		
		  
	-1 or --1M
	
		the input vector size will be  1M = 2^19
		
	-2 or --2M
	
		the input vector size will be  2M = 2^20
		
	-3 or --4M
	
		the input vector size will be  4M = 2^21
		
	-4 or --8M
	
		the input vector size will be  8M = 2^22
		
	-5 or --16M
	
		the input vector size will be  16M = 2^23
		
	-6 or --32M
	
		the input vector size will be  32M = 2^24
	
[--size SIZE]:
	
	CUDA-Quicksort is performed on vectors of the size indicated by 'SIZE'. 
	This flags work only with testGPU and validate test, in compare test the size flags are "skipped"
		
[--device identifier]: 
	Use this option to choose which GPU should be used. Default: --device 0

