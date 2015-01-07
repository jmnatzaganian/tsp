/* world.h

   Author        : James Mnatzaganian
   Date Created  : 11/07/14
   
   Description   : Header file for dealing with the 2D world.
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/

#ifndef __WORLD_H__
#define __WORLD_H__

// Native Includes
#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
////////// Shared
///////////////////////////////////////////////////////////////////////////////

struct City
{
	/*
		Stores the location of a city
	*/
	
	int x, y;
};

typedef struct World
{
	/*
		2D world for the TSP
	*/
	
	int width, height; // World bounds
	int num_cities;    // Number of cities
	City* cities;      // Pointer to array of all of the cities
	float fitness;     // The current fitness
	float fit_prob;    // The fitness probability

	inline __host__ void calc_fitness()
	{
		/*
			Evaluates the fitness function
		*/

		float distance = 0.0;
		for (int i=0; i<num_cities-1; i++)
			distance += (cities[i].x - cities[i + 1].x) * (cities[i].x -      \
				cities[i +1 ].x) + (cities[i].y - cities[i + 1].y)     *      \
				(cities[i].y - cities[i + 1].y);
		fitness = (width * height) / distance;
	}

	inline __host__ float calc_distance()
	{
		/*
			Calculates the distance travelled
		*/

		float distance = 0.0;
		for (int i=0; i<num_cities-1; i++)
			distance += (float)sqrt((float)((cities[i].x - cities[i + 1].x) * \
				(cities[i].x - cities[i + 1].x) + (cities[i].y              - \
				cities[i +1 ].y) * (cities[i].y - cities[i + 1].y)));
		return distance;
	}
} World;

/*
	Makes a new world struct
	
	world      : Pointer to the world to create
	width      : The width of the world
	height     : The height of the world
	num_cities : The number of cities in the world
	seed       : The random seed to use to select the cities
*/
void make_world(World* world, int width, int height, int num_cities, int seed);

///////////////////////////////////////////////////////////////////////////////
////////// CPU functions
///////////////////////////////////////////////////////////////////////////////

/*
	Initialize a world struct
	
	world      : The world to initialize
	width      : The width of the world
	height     : The height of the world
	num_cities : The number of cities in the world
*/
void init_world(World* world, int width, int height, int num_cities);

/*
	Clones one more cities in host memory
	
	src        : Pointer to source cities
	dst        : Pointer to destination cities
	num_cities : The number of cities to clone
*/
void clone_city(City* src, City* dst, int num_cities);

/*
	Clones a single world in host memory
	
	src : Pointer to source world
	dst : Pointer to destination world
*/
void clone_world(World* src, World* dst);

/*
	Frees the world from host memory
	
	world : The world to delete
*/
void free_world(World* world);

/*
	Frees a population (multiple worlds) from host memory
	
	pop      : Pointer to worlds
	pop_size : The number of worlds
*/
void free_population(World* pop, int pop_size);

///////////////////////////////////////////////////////////////////////////////
////////// GPU functions
///////////////////////////////////////////////////////////////////////////////

/*
	Initialize a world struct in device memory
	
	d_world    : The world to initialize on the device
	h_world    : The host world to use as a template

	returns true if an error occurred
*/
bool g_init_world(World* d_world, World* h_world);

/*
	Clone most of the world in device memory
		
	d_world    : The world to initialize on the device
	h_world    : The host world to use as a template

	returns true if an error occurred
*/
bool g_soft_clone_world(World* d_world, World* h_world);

#endif