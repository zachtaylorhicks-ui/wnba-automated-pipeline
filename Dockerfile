
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container to /app
WORKDIR /app

# Install system dependencies needed by some Python libraries (e.g., for building wheels)
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at the root of the WORKDIR
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code from the host to the container's WORKDIR
# This includes the src/, data/, etc. folders
COPY . .

# Specify a default command to run when the container starts.
# We will override this in our GitHub Actions workflow to run specific scripts,
# but it's good practice to have a default.
CMD ["python", "src/predict.py"]
