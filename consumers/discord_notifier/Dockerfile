FROM python:3.12-slim

RUN pip install --no-cache-dir redis

WORKDIR /consumer

COPY common/ ./common/
COPY consumers/discord_notifier/consumer.py ./

ENV PYTHONPATH=/consumer

CMD ["python", "consumer.py"]
