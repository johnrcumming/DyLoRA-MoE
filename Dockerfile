# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define the command to run the training script
CMD ["python", "train.py"]