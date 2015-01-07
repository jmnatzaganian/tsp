/* world.h

   Author        : James Mnatzaganian
   Date Created  : 11/07/14
   
   Description   : Module for dealing with the TSP's world
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/

// Native Includes
#include <set>
#include <tuple>
#include <random>
#include <functional>

// Program Includes
#include "world.h"
#include "common.h"
#include "ga_gpu.h"

///////////////////////////////////////////////////////////////////////////////
////////// Shared functions
///////////////////////////////////////////////////////////////////////////////

void make_world(World* world, int width, int height, int num_cities, int seed)
{
	/*
		Makes a new world struct
		
		world      : Pointer to the world to create
		width      : The width of the world
		height     : The height of the world
		num_cities : The number of cities in the world
		seed       : The random seed to use to select the cities
	*/
	
	// Random number generation
	mt19937::result_type rseed = seed;
	auto rgen = bind(uniform_real_distribution<>(0, 1), mt19937(rseed));
	
	// Initialize the world
	init_world(world, width, height, num_cities);
	
	// Create a set to deal with uniqueness
	set<tuple<int, int>> coordinates;
	set<tuple<int, int>>::iterator it;
	pair<set<tuple<int, int>>::iterator,bool> ret;
	
	// Create some unique random cities
	for (int i=0; i<num_cities; i++)
	{
		while (true)
		{
			// Try to add a new set of coordinates
			tuple<int,int> coors((int)(rgen() * width), \
				(int)(rgen() * height));
			ret = coordinates.insert(coors);
			
			// Break if the city was added successfully
			if (ret.second)
				break;
		}
	}
	
	// Add those cities to the world
	{
		int i = 0;
		for (it=coordinates.begin(); it!=coordinates.end(); it++)
		{
			world->cities[i].x = get<0>(*it);
			world->cities[i].y = get<1>(*it);
			i++;
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
////////// CPU functions
///////////////////////////////////////////////////////////////////////////////

void init_world(World* world, int width, int height, int num_cities)
{
	/*
		Initialize a world struct in host memory
		
		world      : The world to initialize
		width      : The width of the world
		height     : The height of the world
		num_cities : The number of cities in the world
	*/
	
	world->width      = width;
	world->height     = height;
	world->num_cities = num_cities;
	world->fitness    = (float)0.0;
	world->fit_prob   = (float)0.0;
	world->cities     = new City[num_cities * sizeof(City)];
}

void clone_city(City* src, City* dst, int num_cities)
{
	/*
		Clones one more cities in host memory
		
		src        : Pointer to source cities
		dst        : Pointer to destination cities
		num_cities : The number of cities to clone
	*/
	
	memcpy(dst, src, num_cities * sizeof(City));
}

void clone_world(World* src, World* dst)
{
	/*
		Clones a single world in host memory
		
		src : Pointer to source world
		dst : Pointer to destination world
	*/
	
	dst->width      = src->width;
	dst->height     = src->height;
	dst->num_cities = src->num_cities;
	dst->fitness    = src->fitness;
	dst->fit_prob   = src->fit_prob;
	clone_city(src->cities, dst->cities, src->num_cities);
}

void free_world(World* world)
{
	/*
		Frees the world from host memory
		
		world : The world to delete
	*/
	
	delete[] world->cities;
	delete[] world;
}

void free_population(World* pop, int pop_size)
{
	/*
		Frees a population (multiple worlds) from host memory
		
		pop      : Pointer to worlds
		pop_size : The number of worlds
	*/
	
	for (int i=0; i<pop_size; i++)
		delete[] pop[i].cities;
	delete[] pop;
}

///////////////////////////////////////////////////////////////////////////////
////////// GPU functions
///////////////////////////////////////////////////////////////////////////////

bool g_init_world(World* d_world, World* h_world)
{
	/*
		Initialize a world struct in device memory
		
		d_world    : The world to initialize on the device
		h_world    : The host world to use as a template

		returns true if an error occurred
	*/
	
	// Error checking
	bool error;
	
	// Soft clone world
	error = g_soft_clone_world(d_world, h_world);
	if (error)
		return true;
	
	// Allocate space for cities on device
	City *d_city;
	error = checkForError(cudaMalloc((void**)&d_city, h_world->num_cities *   \
		sizeof(City)));
	if (error)
	{
		cout << "DEVICE ERROR - Allocating cities on device during "          \
			<< "world initialization" << endl; return true;
	}
	
	// Update pointer on device
	error = checkForError(cudaMemcpy(&d_world->cities, &d_city,               \
		sizeof(City*), cudaMemcpyHostToDevice));
	if (error)
	{
		cout << "DEVICE ERROR - Updating city pointer on device during "      \
			<< "world initialization" << endl; return true;
	}
	return false;
}

bool g_soft_clone_world(World* d_world, World* h_world)
{
	/*
		Clone most of the world in device memory
		
		d_world    : The world to initialize on the device
		h_world    : The host world to use as a template

		returns true if an error occurred
	*/
	
	// Error checking
	bool error;
	
	error = checkForError(cudaMemcpy(&d_world->width, &h_world->width,        \
		sizeof(int), cudaMemcpyHostToDevice));
	if (error)
	{
		cout << "DEVICE ERROR - Copying world width to device during "        \
			<< "world initialization" << endl; return true;
	}
	error = checkForError(cudaMemcpy(&d_world->height, &h_world->height,      \
		sizeof(int), cudaMemcpyHostToDevice));
	if (error)
	{
		cout << "DEVICE ERROR - Copying world height to device during "       \
			<< "world initialization" << endl; return true;
	}
	error = checkForError(cudaMemcpy(&d_world->num_cities,                    \
		&h_world->num_cities, sizeof(int), cudaMemcpyHostToDevice));
	if (error)
	{
		cout << "DEVICE ERROR - Copying number of cities to device "          \
			<< "during world initialization" << endl; return true;
	}
	return false;
}