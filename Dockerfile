# Use the official Python image as the base image
FROM python:3.9

# Set the working directory in the containe
WORKDIR /app

# Install necessary system packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt 

# Copy the rest of the application code into the container at /app
COPY . .

# Expose port 8000 to the outside world
EXPOSE 8000

# Set the command to run the application
CMD ["uvicorn", "running:app", "--host", "0.0.0.0", "--port", "8000"]