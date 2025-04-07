# set base image (host OS)
FROM python:3.10

# set the working directory in the container
WORKDIR /app

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# Copy code to the working directory
RUN mkdir src
RUN mkdir output
RUN mkdir data
COPY src/* src/
COPY data/* data/

# command to run on container start
ENTRYPOINT ["python", "./src/entrypoint.py"]
