# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY dylo_moe/ /app/dylo_moe/
COPY data/ /app/data/
COPY train.py /app/
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install -r /app/requirements.txt

# Define the command to run the training script
CMD ["python", "train.py"]

CMD [ \
    "python", "train.py", \
    "--datasets", "code_alpaca,mbpp,evol_instruct,code_feedback", \
    "--bf16", \
    "--num_epochs", "10", \
    "--num_experts", "2", \
    "--balance_coefficient", "0.01",  \
    "--cosine_restarts",  \
    "--train_batch_size", "2", \
    "--eval_batch_size", "2", \
    "--gradient_accumulation_steps", "64", \
    "--early_stopping_patience", "5", \
]