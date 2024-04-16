FROM python:3.9.12

COPY requirements.txt requirements.txt

RUN set -eux && pip install --upgrade pip && \pip install --no-cache-dir -r requirements.txt

CMD ["tail", "-f", "/dev/null"]