#!/bin/bash

# Create necessary directories
mkdir -p models/checkpoints
mkdir -p data/corpus
mkdir -p data/db
mkdir -p data/cache
mkdir -p data/sessions

echo "--- Directory Structure Ready ---"

# Note: Automatic download requires 'huggingface-cli'
# You can install it via: pip install huggingface-hub

echo "To download the required models, run the following commands:"
echo ""
echo "1. MedGemma 1.5-4B (Instruction Tuned):"
echo "huggingface-cli download google/medgemma-1.5-4b-it --local-dir models/checkpoints/medgemma-1.5-4b-it"
echo ""
echo "2. MedEmbed (Large):"
echo "huggingface-cli download abhinand/MedEmbed-large-v0.1 --local-dir models/checkpoints/MedEmbed-large-v0.1"
echo ""
echo "3. MedCPT Cross-Encoder:"
echo "huggingface-cli download ncbi/MedCPT-Cross-Encoder --local-dir models/checkpoints/MedCPT-Cross-Encoder"
echo ""
echo "4. EyeCLIP Visual Weights (Manual Download Required):"
echo "Download 'eyeclip_visual_new.pt' from the original repo: https://github.com/Michi-3000/EyeCLIP"
echo "Place it in models/checkpoints/"
echo ""
echo "--- Setup Complete ---"
