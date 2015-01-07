/* common.h

   Author        : James Mnatzaganian
   Date Created  : 11/07/14
   
   Description   : Header for all shared functions
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/

#ifndef __COMMON_H__
#define __COMMON_H__

// Native Includes
#include <ctime>
#include <random>
#include <functional>

// Program Includes
#include "world.h"
#include "log.h"

/* 
	Check for a CUDA error
		
	error : The error to check
		
	returns true for error and false for no error
*/
bool checkForError(cudaError_t error);

/*
	Stops a clocks timer and returns the elapsed time in ms.
	
	clk - The clk to work with.
*/
float end_clock(clock_t clk);

/*
	Initialize the population in host memory

	world    : The world to use
	pop      : The population to create
	pop_size : The number of elements in the population
	seed     : Seed for random number generation

	returns true if an error occurred
*/
void initialize(World* world, World* pop, int pop_size, int seed);

/*
	Updates the generation and global best leaders
	
	pop               : The population to select from
	pop_size          : The number of elements in the population
	generation_leader : The world with the max fitness for this generation
	best_leader       : The world with the best global fitness across all
		generations

	return 1 if this generation is the best, else 0
*/
int select_leader(World* pop, int pop_size, World* generation_leader,
	World* best_leader);

#endif