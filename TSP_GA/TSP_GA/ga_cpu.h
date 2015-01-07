/* ga_cpu.h

   Author        : James Mnatzaganian
   Date Created  : 11/07/14
   
   Description   : Header for GA implementation for the CPU
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/

#ifndef __GA_CPU_H__
#define __GA_CPU_H__

// Program includes
#include "world.h"
#include "log.h"

/*
	Evaluate the fitness function and calculate the
	fitness probabilities.
	
	pop      : The population to create
	pop_size : The number of elements in the population
*/
void evaluate(World* pop, int pop_size);

/*
	Perform the selection algorithm on the CPU.
	This selection algorithm uses Roulette Wheel Selection.
	Two parents will be selected at a time, from the population.

	pop       : The population to select from
	pop_size  : The number of elements in the population
	parents   : The cities for two worlds
	rand_nums : The random numbers to use
*/
void selection(World* pop, int pop_size, City** parents, float* rand_nums);

/*
	Perform the crossover algorithm on the CPU.
	This crossover algorithm uses the Single Point Crossover method.
	
	parents    : The cities for two worlds
	child      : The child to create
	num_cities : The number of cities in the world
	cross_over : The location to perform crossover
*/
void crossover(City** parents, City* child, int num_cities, int cross_over);

/*
	Perform the mutation algorithm on the CPU.
	This mutation algorithm uses the order changing permutation method.
	
	child      : The child to mutate
	rand_nums  : The random numbers to use
*/
void mutate(City* child, int* rand_nums);

/*
	Runs the genetic algorithm on the CPU.
		
	prob_mutation  : The probability of a mutation occurring
	prob_crossover : The probability of a crossover occurring
	pop_size       : The number of elements in the population
	max_gen        : The number of generations to run for
	world          : The seed world, containing all of the desired cities
	gen_log        : A pointer a logger to be used for logging the
		generation statistics
	seed           : Seed for all random numbers
*/
void execute(float prob_mutation, float prob_crossover, int pop_size,
	int max_gen, World* world, logger* gen_log, int seed);

#endif