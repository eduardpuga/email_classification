# Dockerfile.app
FROM python:3.9

# Instalar wait-for-it
RUN apt-get update && apt-get install -y wait-for-it

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de requerimientos y el script principal
COPY app/requirements.txt requirements.txt
COPY app/main.py main.py
COPY app/train_model.py train_model.py
COPY app/utils.py utils.py

# Instalar las dependencias
RUN pip install -r requirements.txt

# Copiar el resto de los archivos
COPY app/ .

# Comando para ejecutar la aplicación
CMD ["python", "main.py"]
