# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Set the working directory in the container
WORKDIR /app

# copy and install requirements
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the current directory contents into the container at /app
COPY dylo_moe/ /app/dylo_moe/
COPY data/ /app/data/
COPY benchmarks/ /app/benchmarks/
COPY train.py /app/
COPY benchmark.py /app/
COPY entrypoint.py /app/

# Set the unified entrypoint
ENTRYPOINT ["python", "entrypoint.py"]

# Default to training with common parameters
CMD ["--train", "--datasets", "code_alpaca,mbpp", "--bf16", "--num_epochs", "5", "--num_experts", "2", "--balance_coefficient", "0.01", "--cosine_restarts"]