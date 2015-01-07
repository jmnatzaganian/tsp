/* ga_gpu.h

   Author        : James Mnatzaganian
   Date Created  : 11/07/14
   
   Description   : Header for GA implementation for the GPU
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/

#ifndef __GA_GPU_H__
#define __GA_GPU_H__

// Native includes
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// Program includes
#include "world.h"
#include "log.h"

/* 
	Check for a CUDA error
	
	error : The error to check
	
	returns true for error and false for no error
*/
bool checkForError(cudaError_t error);

/* 
	Check for a kernel error
	
	err_msg : The error message to print
	
	returns true for error and false for no error
*/
bool checkForKernelError(const char *err_msg);

/*
	Get the thread id in a 2D grid with 2D blocks
*/
__device__ int getGlobalIdx_2D_2D();

/*
	Perform crossover on the device

	old_pop   : The old population (where the parents are located)
	new_pop   : The new population (where the children will be)
	sel_ix    : The indexes of the parents in the old population
	cross_loc : The crossover locations
	tid       : The thread id
*/
__device__ void crossover(World* old_pop, World* new_pop, int* sel_ix,        \
	int* cross_loc, int tid);

/*
	Perform mutation on the device
		
	new_pop    : The new population (where the children will be)
	mutate_loc : The crossover locations
	tid        : The thread id
*/
__device__ void mutate(World* new_pop, int* mutate_loc, int tid);

/*
	Kernel for evaluating the fitness function

	pop      : The population
	pop_size : The number of elements in the population
*/
__global__ void fitness_kernel(World* pop, int pop_size);

/*
	Kernel for evaluating the partial probabilities used for selection

	pop      : The population to create
	pop_size : The number of elements in the population
	fit_sum  : The sum of all fitnesses
*/
__global__ void fit_sum_kernel(World* pop, int pop_size, float* fit_sum);

/*
	Kernel for evaluating the probabilities used for selection

	pop      : The population
	pop_size : The number of elements in the population
	fit_sum  : The sum of all fitnesses
*/
__global__ void fit_prob_kernel(World* pop, int pop_size, float* fit_sum);

/*
	Kernel for finding the max fitness

	pop      : The population
	pop_size : The number of elements in the population
	max      : The max fitness
	ix       : The index at which the max fitness was found
	*/
__global__ void max_fit_kernel(World* pop, int pop_size, float* max, int* ix);

/*
	Kernel for finding the indexes of the selected parents

	pop       : The population
	pop_size  : The number of elements in the population
	rand_nums : The random numbers to use
	sel_ix    : The indexes of the parents in the population
*/
__global__ void selection_kernel(World* pop, int pop_size, float* rand_nums,  \
	int* sel_ix);

/*
	Kernel for creating the children for the new population

	old_pop        : The old population (where the parents are located)
	new_pop        : The new population (where the children will be)
	pop_size       : The number of elements in the population
	sel_ix         : The indexes of the parents in the old population
	prob_crossover : The probability of corssover occuring
	prob_cross     : The probabilities of crossover occuring
	cross_loc      : The crossover locations
	prob_mutation  : The probability of mutation occuring
	prob_mutate    : The probabilities of mutation occuring
	mutate_loc     : The mutation locations
*/
__global__ void child_kernel(World* old_pop, World* new_pop, int pop_size,    \
	int* sel_ix, float prob_crossover, float* prob_cross, int* cross_loc,     \
	float prob_mutation, float* prob_mutate, int* mutate_loc);

/*
	Initialize the population in device memory

	world    : The world to use
	pop      : The population to create
	pop_size : The number of elements in the population
	seed     : Seed for random number generation

	returns true if an error occurred
*/
bool g_initialize(World* world, World* pop, int pop_size, int seed);

/*
	Performs evaluation on the GPU

	pop      : The population to evaluate
	pop_size : The size of the population
	Block    : CUDA block definition
	Grid     : CUDA grid definition

	returns true if an error occurs
*/
bool g_evaluate(World *pop, int pop_size, dim3 Block, dim3 Grid);

/*
	Updates the generation and global best leaders
	
	pop               : The population to select from
	pop_size          : The number of elements in the population
	generation_leader : The world with the max fitness for this generation
	best_leader       : The world with the best global fitness across all
		generations
	Block    : CUDA block definition
	Grid     : CUDA grid definition

	return 1 if this generation is the best, 0 if not, and -1 for error
*/
int g_select_leader(World* pop, int pop_size, World* generation_leader,
	World* best_leader, dim3 Block, dim3 Grid);

/*
	Runs the genetic algorithm on the GPU.
		
	prob_mutation  : The probability of a mutation occurring
	prob_crossover : The probability of a crossover occurring
	pop_size       : The number of elements in the population
	max_gen        : The number of generations to run for
	world          : The seed world, containing all of the desired cities
	gen_log        : A pointer a logger to be used for logging the
		generation statistics
	seed           : Seed for all random numbers

	returns true if an error occurs
*/
bool g_execute(float prob_mutation, float prob_crossover, int pop_size,
	int max_gen, World* world, logger* gen_log, int seed);

#endif