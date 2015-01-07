# tsp
## Intro
This is a C++ and CUDA implementation for solving the Traveling Salesman
Problem (TSP) using a Genetic Algorithm (GA). Details of the implementation
may be found [here](http://techtorials.me/the-traveling-salesman-problem-genetic-algorithm-in-c-and-cuda/).

This code was developed in Visual Studio 2010; however, it should work in other
environments and be cross-platform, assuming the appropriate libraries and
compilers are used.

## Prerequisites
### C++ Prerequisites
- A C++ compiler (Visual Studio 2010)

### CUDA Prerequisites
- 64bit 5.0 version of the [NVIDIA's CUDA Toolkit](https://developer.nvidia.com/
cuda-toolkit-50-archive) (newer versions should work - 64bit is strongly
recommended, but not necessarily required)

- An NVIDIA capable GPU (this code was designed for a GTX-480)

### Video Rendering Prerequisites
- [ffmpeg](https://www.ffmpeg.org/download.html)

- [Python 2.7.8](https://www.python.org/download/releases/2.7.8/)

- [numpy](http://www.numpy.org/)

- [matplotlib](http://matplotlib.org/)

- [networkx](https://networkx.github.io/)

## Usage
The primary code is located in "TSP_GA". A full Visual Studio 2010 solution was
provided for your convenience. You may simply open that solution and proceed,
assuming the CUDA toolkit has been installed and configured properly.

Some paths have been hard-coded. You'll need change line 75 in "main.cpp" to be
an existing directory where data may be outputted. You can disable the GPU
portion of the code by simply commenting out lines 155 - 199.

For clarity purposes, all CPU-specific code follows a file naming convention
ending in "_cpu"; similarly, GPU-specific code has a "_gpu" postfix.

The file "tsp_graph" is used to to generate a video for each generation. Static
images are created using NetworkX and ffmpeg is used to compile those images
into a video. To use this code, lines 225 - 243 will need to be edited. Make
sure that "main" is being called with correct file paths (refer to its
documentation for more details). Additionally, make sure that lines 242 and 243
are utilizing the correct paths. The first path should be the full path to
"ffmpeg.exe". The second path should be the full path to the directory
containing the images created by "main". The third path should be the full path
to where the output video should be created.
## Author
The original author of this code was James Mnatzaganian. For contact info, as
well as other details, see his corresponding [website](http://techtorials.me).
## Legal
This code is licensed under the [MIT license](http://opensource.org/licenses/
mit-license.php).