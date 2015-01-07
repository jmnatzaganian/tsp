/* tsp_ga.cpp

   Author        : James Mnatzaganian
   Date Created  : 11/07/14
   
   Description   : Solve the TSP problem using a GA
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/

// Native Includes
#include <ctime>
#include <iostream>
#include <string.h>
#include <sstream>

// Program Includes
#include "common.h"
#include "log.h"
#include "ga_cpu.h"
#include "ga_gpu.h"

using namespace std;

int main()
{
	/*
		Entry point for the program.
		
		Performs a few iterations of the TSP on the CPU and GPU.
	*/
	
	// Logger
	logger* gen_log = new logger;

	// GA parameters
	float prob_mutation  = (float)0.15; // The probability of a mutation
	float prob_crossover = (float)0.8;  // The probability of a crossover
	int world_seed       = 12345678;    // Seed for initial city selection
	int ga_seed          = 87654321;    // Seed for all other random numbers
	
	// World parameters
	int world_width  = 10000; // Width of the world
	int world_height = 10000; // Height of the world
	
	// The test cases
	int iterations          = 1;  // Number of full runs
	const int num_cases     = 1; // How many trials to test
	int cases[num_cases][3] =     // num_cities, pop_size, max_gen
	{
		//{25, 100,    1000},
		//{25, 1000,   1000},
		//{25, 10000,  100},
		{25, 100000, 10}/*,

		{50, 100,    1000},
		{50, 1000,   1000},
		{50, 10000,  100},
		{50, 100000, 10},

		{100, 100,    1000},
		{100, 1000,   1000},
		{100, 10000,  100},
		{100, 100000, 10},

		{250, 100,    1000},
		{250, 1000,   1000},
		{250, 10000,  100},
		{250, 100000, 10}*/
	};
	
	// Timing
	clock_t iter_time, total_time;
	
	// The output path
	char path[100];
	strcpy_s(path, "E:\\HPA\\TSP\\Results\\");
	
	// Loop over all city combinations
	for (int i=0; i<num_cases; i++)
	{
		// GA params
		int num_cities = cases[i][0];
		int pop_size   = cases[i][1];
		int max_gen    = cases[i][2];

		// Generate new strings for the output
		char c_timing_path[100];
		char c_gen_path[100];
		char c_stats_path[100];
		char g_timing_path[100];
		char g_gen_path[100];
		char g_stats_path[100];
		char c0[100];
		char c1[100];
		char c2[100];
		char g0[100];
		char g1[100];
		char g2[100];
		sprintf_s(c0, "%d_%d-cpu_gen.csv", num_cities, pop_size);
		sprintf_s(c1, "%d_%d-cpu_timing.csv", num_cities, pop_size);
		sprintf_s(c2, "%d_%d-cpu_stats.csv", num_cities, pop_size);
		sprintf_s(g0, "%d_%d-gpu_gen.csv", num_cities, pop_size);
		sprintf_s(g1, "%d_%d-gpu_timing.csv", num_cities, pop_size);
		sprintf_s(g2, "%d_%d-gpu_stats.csv", num_cities, pop_size);

		// Make the world
		World* world = new World[sizeof(World)];
		make_world(world, world_width, world_height, num_cities, world_seed);
		
		// Build the strings for logging purposes
		strcpy_s(c_timing_path, path);
		strcat_s(c_timing_path, c0);
		strcpy_s(c_gen_path, path);
		strcat_s(c_gen_path, c1);
		strcpy_s(c_stats_path, path);
		strcat_s(c_stats_path, c2);
		strcpy_s(g_timing_path, path);
		strcat_s(g_timing_path, g0);
		strcpy_s(g_gen_path, path);
		strcat_s(g_gen_path, g1);
		strcpy_s(g_stats_path, path);
		strcat_s(g_stats_path, g2);
		
		cout << endl << "###################################################" \
			<< "############################" << endl;
		cout << "##### CPU - START" << endl;
		cout << "###########################################################" \
			<< "####################" << endl << endl;

		// CPU timing
		gen_log->start(c_gen_path, c_timing_path, c_stats_path);
		total_time = clock();
		for (int j=0; j<iterations; j++)
		{
			iter_time = clock();
			execute(prob_mutation, prob_crossover, pop_size, max_gen, world,  \
				gen_log, ga_seed);
			gen_log->write_stats(j + 1, "CPU", end_clock(iter_time),          \
				prob_mutation, prob_crossover, pop_size, max_gen, world_seed, \
				ga_seed, world_width, world_height, num_cities);
		}
		gen_log->write_stats(-1, "CPU", end_clock(total_time), prob_mutation, \
			prob_crossover, pop_size, max_gen, world_seed, ga_seed,           \
			world_width, world_height, num_cities);
		gen_log->end();

		cout << endl << "###################################################" \
			<< "############################" << endl;
		cout << "CPU - END" << endl;
		cout << "###########################################################" \
			<< "####################" << endl << endl;

		cout << "===========================================================" \
			<< "====================" << endl << endl;

		cout << "###########################################################" \
			<< "####################" << endl;
		cout << "GPU - START" << endl;
		cout << "###########################################################" \
			<< "####################" << endl << endl;

		// Clear out the device's memory
		cudaDeviceReset();
		
		// GPU warmup pass - A single generation should be good enough
		gen_log->start(g_gen_path, g_timing_path, g_stats_path);
		g_execute(prob_mutation, prob_crossover, pop_size, 1,                 \
				world, gen_log, ga_seed);
		gen_log->end();

		// Clear out the device's memory
		cudaDeviceReset();

		// GPU timing
		gen_log->start(g_gen_path, g_timing_path, g_stats_path);
		total_time = clock();
		for (int j=0; j<iterations; j++)
		{
			cudaDeviceReset(); // Clear

			iter_time = clock();
			g_execute(prob_mutation, prob_crossover, pop_size, max_gen,       \
				world, gen_log, ga_seed);
			gen_log->write_stats(j + 1, "GPU", end_clock(iter_time),          \
				prob_mutation, prob_crossover, pop_size, max_gen, world_seed, \
				ga_seed, world_width, world_height, num_cities);
		}
		gen_log->write_stats(-1, "GPU", end_clock(total_time), prob_mutation, \
			prob_crossover, pop_size, max_gen, world_seed, ga_seed,           \
			world_width, world_height, num_cities);
		gen_log->end();

		// Clear out the device's memory
		cudaDeviceReset();

		cout << endl << "###################################################" \
			<< "############################" << endl;
		cout << "GPU - END" << endl;
		cout << "###########################################################" \
			<< "####################" << endl << endl;

		// Cleanup
		free_world(world);
	}
	
	delete gen_log;
	
	// Success
	return 0;
}