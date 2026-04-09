FROM python:3.12-slim

RUN pip install --no-cache-dir redis

WORKDIR /skill

# Copy shared SDK
COPY common/ ./common/

# Copy skill source
COPY skills/package_delivery_alert/config.py \
     skills/package_delivery_alert/detector.py \
     skills/package_delivery_alert/skill.py ./

# common/ is a sibling directory; add /skill to PYTHONPATH so
# "from common.event_bus import EventBus" resolves correctly.
ENV PYTHONPATH=/skill

CMD ["python", "skill.py"]
