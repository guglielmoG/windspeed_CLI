FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY . .

RUN ./setup.sh

CMD ["python", "./utils.py"]