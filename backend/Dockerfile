# Use an official Python runtime as a parent image, which includes a Debian base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install C++ dependencies (CMake, GCC, OpenMP, OpenMPI)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libomp-dev \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the entire project into the container
COPY . /app/

# Build the C++ Graph Engine
RUN mkdir -p build && cd build && cmake .. && make -j4

# Install Python Backend Dependencies
WORKDIR /app/backend
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
