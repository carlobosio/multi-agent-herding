# Use an official PyTorch image as the base
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy your requirements file if you have one
COPY requirements.txt .

# Install any additional dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Run your application
CMD ["python", "main.py"]