FROM python:3.12-slim

WORKDIR /skill

# No extra dependencies beyond the stdlib — copy source only.
COPY config.py detector.py notifier.py skill.py ./

CMD ["python", "skill.py"]
