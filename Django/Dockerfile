FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir Django==5.2.6

COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
