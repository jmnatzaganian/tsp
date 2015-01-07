/* log.cpp

   Author        : James Mnatzaganian
   Contact       : http://techtorials.me
   Date Created  : 11/07/14
   
   Description   : Module for handling all logging operations
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/

// Native includes
#include <iostream>

// Program includes
#include "log.h"

using namespace std;

void print_status(World* generation_leader, World* best_leader, int generation)
{
	/*
		Prints the current status to stdout
		
		generation_leader : The leader for the current generation
		best_leader       : The leader out of all generations
		generation        : The generation index
	*/
	
	cout << "Generation " << generation << ":" << endl;
	cout << "  Generation Leader's Fitness: "  << generation_leader->fitness \
		<< endl;
	cout << "  Best Leader's Fitness      : "  << best_leader->fitness << endl;
}