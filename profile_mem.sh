#!/bin/bash

# Run the Python script
# The script will run until you interrupt it with Ctrl+C
# MEMRAY_FOLLOW_FORK=1 python TA_autoencoder/train_autoencoder.py
MEMRAY_FOLLOW_FORK=1 python mem_profile_env.py

# After interruption, generate memray visualizations
if [ -f "env_profile.bin" ]; then
    echo "Generating flamegraph"
    memray flamegraph env_profile.bin -o comprehensive_flame.html
    
    # echo "Generating table"
    # memray table env_profile.bin -o comprehensive_table.txt
    
    # echo "Generating summary"
    # memray summary comprehensive_profile.bin -o comprehensive_summary.txt

    # echo "Generating tree view"
    # memray tree comprehensive_profile.bin -o comprehensive_tree.txt
else
    echo "No comprehensive_profile.bin file found. Make sure the Python script completed successfully."
fi

echo "Profiling complete. Check the generated HTML and TXT files for analysis."