FROM python:3.10-slim-buster
WORKDIR /app
COPY . /app/

RUN apt update -y && apt upgrade -y
RUN pip install -r requirements.txt
CMD ["python", "app.py"]