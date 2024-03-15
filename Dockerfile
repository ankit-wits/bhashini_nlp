# Use the official Python image as a base image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the IndicTransTokenizer directory into the container at /app
COPY ./ /app/IndicTransTokenizer

# Install the IndicTransTokenizer package
RUN pip install --editable ./IndicTransTokenizer

# Copy the rest of the application code into the container at /app
COPY . /app

# Expose port 80 to the outside world
EXPOSE 80

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
