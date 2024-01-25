# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Make a directory named 'front' in the image
RUN mkdir /front

# Change the working directory to '/front'
WORKDIR /front

# Install the necessary libraries
RUN pip install requests numpy pandas matplotlib scikit-learn streamlit

# Copy the contents from your current directory (.) to the '/front' directory in the image
COPY . /front

# Inform Docker that the container listens on the specified network port at runtime
EXPOSE 8501

# The command that will be executed when the Docker container starts
CMD ["streamlit", "run", "f_main.py"]
