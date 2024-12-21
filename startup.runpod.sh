#!/bin/bash
set -e

cd /app

# Verify installation
export CXX=/usr/local/bin/gxx-wrapper
python3.11 example.py

# Set compiler wrapper for runtime
export CXX=/usr/local/bin/gxx-wrapper

echo "Launching RunPod handler..."
python3.11 -u rp_handler.py 