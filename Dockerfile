# Use python base image: slim-buster (lightweight Python 3.10 base image)
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into the container
COPY . .

# Set the command to run the main script
CMD ["python", "main.py"]