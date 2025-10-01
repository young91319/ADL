#!/bin/bash

# Download script for ADL HW1
# Downloads data and model checkpoints from Google Drive

echo "Starting download process..."
echo "============================="

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Error: gdown is not installed. Please install it with: pip install gdown"
    exit 1
fi

# Download data
echo "📁 Downloading data folder..."
if gdown --folder https://drive.google.com/drive/u/0/folders/1ZODIS-1qqCprZafnqkuQDBSIr7HoZSaK; then
    echo "✅ Data download completed successfully"
else
    echo "❌ Data download failed"
    exit 1
fi

# Download model checkpoints
echo "🤖 Downloading model checkpoints..."
if gdown --folder https://drive.google.com/drive/u/0/folders/1QI1jz0Ury6FqtCPfiKEJ_mptmL0SDWz3; then
    echo "✅ Model checkpoints download completed successfully"
else
    echo "❌ Model checkpoints download failed"
    exit 1
fi

echo "============================="
echo "🎉 All downloads completed successfully!"
echo "📁 Check the downloaded folders in your current directory"