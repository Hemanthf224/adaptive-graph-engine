#!/bin/bash

# Exit on error
set -e

echo "========================================="
echo " Adaptive Graph Engine - Dataset Downloader"
echo "========================================="

# Ensure we are in the project root
cd "$(dirname "$0")/.."
mkdir -p data
cd data

# Function to download and extract a dataset
download_dataset() {
    URL=$1
    FILE=$(basename "$URL")
    TXT_FILE="${FILE%.gz}"
    
    echo "-----------------------------------------"
    if [ -f "$TXT_FILE" ]; then
        echo "Dataset $TXT_FILE already exists. Skipping."
    else
        echo "Downloading $FILE from SNAP..."
        wget -q --show-progress "$URL"
        
        echo "Extracting $FILE..."
        gunzip "$FILE"
        
        echo "Successfully extracted $TXT_FILE"
    fi
}

# Download Datasets
download_dataset "https://snap.stanford.edu/data/web-Google.txt.gz"
download_dataset "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz"
download_dataset "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz"

echo "========================================="
echo " All datasets successfully downloaded!"
echo "========================================="
