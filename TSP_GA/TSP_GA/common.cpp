/* common.cpp

   Author        : James Mnatzaganian
   Date Created  : 11/07/14
   
   Description   : Module for implementing functions shared between the CPU and
   GPU implementations.
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/

// Native includes
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <algorithm>

// Program includes
#include "common.h"

using namespace std;

bool checkForError(cudaError_t error)
{
	/* 
		Check for a CUDA error
		
		error : The error to check
		
		returns true for error and false for no error
	*/

	if (error != cudaSuccess)
	{
		cout << cudaGetErrorString(error) << endl;
		return true;
	}
	else
	{
		return false;
	}
}

float end_clock(clock_t clk)
{
	/*
		Stops a clocks timer and returns the elapsed time in ms.
		
		clk - The clk to work with.
	*/
	
	return ((float)((clock() - clk) * 1000) / (float)CLOCKS_PER_SEC);
}

void initialize(World* world, World* pop, int pop_size, int seed)
{
	/*
		Initialize the population in host memory

		world    : The world to use
		pop      : The population to create
		pop_size : The number of elements in the population
		seed     : Seed for random number generation
	*/

	// Set the seed for random number generation
	srand(seed);

	for (int i=0; i<pop_size; i++)
	{
		// Clone world
		pop[i].cities = new City[world->num_cities * sizeof(City)];
		clone_world(world, &pop[i]);

		// Randomly adjust the path between cities
		random_shuffle(&pop[i].cities[0], &pop[i].cities[world->num_cities]);
	}
}

int select_leader(World* pop, int pop_size, World* generation_leader,
	World* best_leader)
{
	/*
		Updates the generation and global best leaders
		
		pop               : The population to select from
		pop_size          : The number of elements in the population
		generation_leader : The world with the max fitness for this generation
		best_leader       : The world with the best global fitness across all
			generations
	
		return 1 if this generation is the best, else 0
	*/

	// Find element with the largest fitness function
	int ix = 0;
	for (int i=1; i<pop_size; i++)
	{
		if (pop[i].fitness > pop[ix].fitness)
			ix = i;
	}

	// Store generation leader
	clone_world(&pop[ix], generation_leader);

	// Update best leader
	if (generation_leader->fitness > best_leader->fitness)
	{
		clone_world(generation_leader, best_leader);
		return 1;
	}

	return 0;
}