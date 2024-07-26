#!/bin/bash

# Run memray to profile memory usage
memray run --native TA_autoencoder/memory_profile_autoencoder.py

# Generate a flamegraph of memory allocation
memray flamegraph memray-results/*.bin -o memory_profile_flame.html

# Generate a table of memory usage
memray table memray-results/*.bin -o memory_profile_table.txt

# Generate a summary of memory usage
memray summary memray-results/*.bin -o memory_profile_summary.txt

# Run the memory_profiler for line-by-line analysis
# python -m memory_profiler TA_autoencoder/memory_profile_autoencoder.py > memory_profile.txt 2>&1

echo "Memory profiling complete. Check memory_profile.png for visualization and memory_profile.txt for line-by-line analysis."