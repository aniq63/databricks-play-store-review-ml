#!/bin/bash

# PlayStore Review ML - AWS EC2 Deployment Script
# This script installs Docker and runs the backend container.

set -e

echo "Updating system..."
sudo dnf update -y

echo "Installing Docker..."
sudo dnf install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

echo "Docker installed successfully. Please log out and log back in for group changes to take effect."
echo "Or run 'newgrp docker' to use it in this session."

# Define image name
IMAGE_NAME="aniqramzan/databricks-reviews:main"

echo "Pulling latest image: $IMAGE_NAME"
sudo docker pull $IMAGE_NAME

echo "Stopping existing container (if any)..."
sudo docker stop backend-api || true
sudo docker rm backend-api || true

echo "Running new container..."
sudo docker run -d \
    --name backend-api \
    -p 80:8000 \
    --restart unless-stopped \
    $IMAGE_NAME

echo "Deployment complete! API is running on port 80."
echo "Check status with: sudo docker ps"
echo "Check logs with: sudo docker logs -f backend-api"
