#!/bin/bash

# Script to comprehensively uninstall Docker on Linux systems

# Uninstall Docker Engine, CLI, Containerd, and Docker Compose packages
sudo apt-get purge -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-compose

# Remove Docker's official GPG key and repository
sudo rm -f /etc/apt/keyrings/docker.gpg
sudo rm -f /etc/apt/sources.list.d/docker.list

# Remove Docker data directories
sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd

# Remove Docker configuration directories
sudo rm -rf ~/.docker
sudo rm -rf /etc/docker

# Clean up any remaining Docker-related packages
sudo apt-get autoremove -y
sudo apt-get autoclean

# Update package lists
sudo apt-get update

echo "Docker has been completely uninstalled from the system."