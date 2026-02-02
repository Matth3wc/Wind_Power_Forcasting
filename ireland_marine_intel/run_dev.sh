#!/bin/bash
# Development startup script for Ireland Marine Intelligence Platform

set -e

echo "🌊 Ireland Marine Intelligence Platform"
echo "======================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create necessary directories
mkdir -p models/saved
mkdir -p data/cache

# Copy environment file if not exists
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env file from template"
fi

# Start the server
echo ""
echo "Starting development server..."
echo "API: http://localhost:8000"
echo "Docs: http://localhost:8000/docs"
echo "Frontend: http://localhost:8000 (or serve frontend/ separately)"
echo ""

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
