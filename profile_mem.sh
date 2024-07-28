#!/bin/bash

# Run the Python script
# The script will run until you interrupt it with Ctrl+C
MEMRAY_FOLLOW_FORK=1 python TA_autoencoder/memory_profile_autoencoder.py

# After interruption, generate memray visualizations
# Use a wildcard to match the generated .bin file(s)
for file in *.bin; do
    if [ -f "$file" ]; then
        base_name=$(basename "$file" .bin)
        
        echo "Generating flamegraph for $file"
        memray flamegraph "$file" -o "${base_name}_flame.html"
        
        echo "Generating table for $file"
        memray table "$file" -o "${base_name}_table.txt"
        
        echo "Generating summary for $file"
        memray summary "$file" -o "${base_name}_summary.txt"

        echo "Generating tree view for $file"
        memray tree "$file" -o "${base_name}_tree.txt"
    fi
done

echo "Profiling complete. Check the generated HTML and TXT files for analysis."