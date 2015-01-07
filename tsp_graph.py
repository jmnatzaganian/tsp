# tsp_graph.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 11/10/14
#	
# Description    : Plot the traveling salesman's route.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2014 James Mnatzaganian

"""
Plot the traveling salesman's route.
"""

__docformat__ = 'epytext'

# Native imports
import csv, os

# Third party imports
try:	
	import numpy as np
except:
	raise('This module requires "numpy".')
try:
	import matplotlib.pyplot as plt
except:
	raise('This module requires "matplotlib".')
try:
	import networkx as nx
except:
	raise('This module requires "networkx".')

def read_route(row):
	"""
	Reads in a single route.
	
	@param row: An iterable object containing the positions in string format.
	The specific format is "x_y". The connections are from the current object
	to the next object.
	
	@return:
		A tuple containing:
			1) A dictionary of nodes and their corresponding positions
			2) A list of node connections
	"""
	
	# Initialize the return values
	positions   = {}
	connections = []
	
	for i, pos in enumerate(row):
		x, y = [int(loc) for loc in pos.split('_')]
		connections.append((i, i + 1))
		positions[i] = (x, y)
	connections.pop()
		
	return positions, connections
	
def read_distances(distance_path):
	"""
		Reads in the distance data.
		
		@param distance_path: Full path to the distance CSV file.
		
		@return:
			A tuple containing:
				1) Distance traveled
				2) Boolean (1 or 0) denoting new best or not
	"""
	
	distance = []
	
	with open(distance_path, 'rb') as f:
		reader = csv.reader(f)
		reader.next()
		row  = reader.next()
		dist = float(row[3])
		distance.append((dist, 1))
		best_distance = dist
		for row in reader:
			dist = float(row[3])
			if dist < best_distance:
				best_distance = dist
				distance.append((dist, 1))
			else:
				distance.append((dist, 0))
				
	return distance

def gen_route(positions, connections):
	"""
		Generate a plot of the salesperson's route.
		
		@param positions: A dictionary of nodes and their corresponding
		positions.
		
		@param connections: A list of the connections between cities.
		
		@return: The current graph object
	"""
	
	# Create a directed graph
	G = nx.Graph()
	
	# Add the nodes
	for n, p in positions.iteritems():
		G.add_node(n, pos=p)
	
	# Add the edges
	for connection in connections:
		G.add_edge(*connection)
	
	return G

def gen_plot(current_G, gen_ix, gen_dist, best_G, best_ix, best_dist,
	out_path, num_cities):
	"""
	Builds and saves the current image.
	
	@param current_G: A graph object for the current generation.
	
	@param gen_ix: The current generation number.
	
	@param gen_dist: The distance for the current generation.
	
	@param best_G: A graph object for the best generation.
	
	@param best_ix: The generation that the best answer was found.
	
	@param best_dist: The distance for the best generation.
	
	@param out_path: The full path to where the image should be created.
	
	@param num_cities: The number of cities in the world.
	"""
	
	# 1x2 grid, first item
	f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
	ax1.set_title('Best Leader\nTraveled {0} m\nGeneration {1}'.format(
		'{:,}'.format(int(best_dist)), best_ix), y=0.91)
	nx.draw(best_G, nx.get_node_attributes(best_G, 'pos'), node_size=50,
		node_color='b', ax=ax1)
	
	# 1x2 grid, second item
	ax2.set_title(u'Generation Leader\nTraveled {0} m\nGeneration {1}' \
		.format('{:,}'.format(int(gen_dist)), gen_ix), y=0.91)
	nx.draw(current_G, nx.get_node_attributes(current_G, 'pos'), node_size=50,
		node_color='b', ax=ax2)
	
	# Save and close the plot
	f.text(0.5, 0.98, '{0} Cities'.format('{:,}'.format(int(num_cities))),
		ha='center', va='top', size='x-large')
	plt.subplots_adjust(left=-0.01, right=1.01, top=0.88, bottom=-0.12)
	plt.savefig(out_path, dpi=240)
	plt.close()
	
def main(route_path, distance_path, out_dir, num_cities):
	"""
	Plot the routes the salesperson took.
	
	@param route_path: Path to a CSV file containing the routes. Each row is a
	route. Each item in the row contains the positions in the format of "x_y".
	
	@param out_dir: The location to save the images.
	
	@param num_cities: The number of cities in the world.
	"""
	
	# Get the distances and leader stats
	distances = read_distances(distance_path)
	
	with open(route_path, 'rb') as f:
		reader   = csv.reader(f)
		out_path = os.path.join(out_dir, 'gen_0.png')
		
		# Initialize the best
		first     = read_route(reader.next())
		current_G = gen_route(*first)
		best_G    = current_G
		best_ix   = 0
		
		gen_plot(current_G, 0, distances[0][0], best_G, best_ix,
			distances[0][0], out_path, num_cities)
		
		for i, row in enumerate(reader, 1):
			out_path  = os.path.join(out_dir, 'gen_{0}.png'.format(i))
			
			current_G = gen_route(*read_route(row))
			if distances[i][1]:
				best_G  = current_G
				best_ix = i
				
			gen_plot(current_G, i, distances[i][0], best_G, best_ix,
				distances[best_ix][0], out_path, num_cities)

if __name__ == '__main__':
	import shutil
	sizes  = [25, 50, 100, 250]
	b_p    = os.path.dirname(os.getcwd())
	ffmpeg = os.path.join(os.getcwd(), 'ffmpeg', 'bin', 'ffmpeg.exe')
	for size in sizes:
		g_p = os.path.join(b_p, 'Results', str(size), 'cpu_gen.csv')
		d_p = os.path.join(b_p, 'Results', str(size), 'cpu_timing.csv')
		o_p = os.path.join(b_p, 'Images', str(size))
		v_p = os.path.join(b_p, 'Videos', 'gen_{0}.mkv'.format(size))
		if os.path.exists(o_p):
			shutil.rmtree(o_p)
		os.makedirs(o_p)
		main(g_p, d_p, o_p, size)
		# if not os.path.exists(os.path.dirname(v_p)):
			# os.makedirs(v_p)
		# elif os.path.exists(v_p):
			# os.remove(v_p)
		os.system('{0} -framerate 10 -i {1} -c:v libx264 {2}'.format(
			ffmpeg,	os.path.join(o_p, 'gen_%d.png'), v_p))