# We start from a base Image 
FROM python:3.8-slim-buster

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install python and pip
RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install jupyter pandas numpy matplotlib scikit-learn tensorflow 

# Copy the Jupyter notebook to the working directory
COPY final_ML.ipynb /app/final_ML.ipynb

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run Jupyter notebook when the container launches
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]
