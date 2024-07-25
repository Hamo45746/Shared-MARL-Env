#!/bin/bash

# Run the memory profiler and generate a profile file
python -m cProfile -o memory_profile.prof TA_autoencoder/memory_profile_autoencoder.py

# Generate a PNG visualization from the profile
gprof2dot -f pstats memory_profile.prof | dot -Tpng -o memory_profile.png

# Run the memory_profiler for line-by-line analysis
# python -m memory_profiler TA_autoencoder/memory_profile_autoencoder.py > memory_profile.txt 2>&1

echo "Memory profiling complete. Check memory_profile.png for visualization and memory_profile.txt for line-by-line analysis."