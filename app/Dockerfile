FROM python:3.9

WORKDIR /app

## CONTINUAR AQUÍ
# Copy the requirements file to the container
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose port 5000 for the Flask application
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "main.py"]