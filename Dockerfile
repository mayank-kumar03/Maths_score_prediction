FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY . /app


# Install the dependencies
RUN apt update -y && apt install awscli -y
RUN pip install  -r requirements.txt



# Start the application
CMD ["python", "app.py"]