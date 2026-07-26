#!/bin/bash

# 1. Start Docker containers in the background
echo "Starting Docker Compose..."
cd scripts && docker compose up -d && cd ..

# 2. Start the Uvicorn backend in the background
echo "Starting Uvicorn backend..."
uvicorn backend.main:app --reload  & 

# 3. Start the frontend
echo "Starting frontend..."
cd frontend && npm run dev

# first give permission so run this command
# chmod +x start.sh
# and then
# ./start.sh

