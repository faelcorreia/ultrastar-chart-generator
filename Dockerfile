FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./requirements.txt /app/

RUN pip install -r requirements.txt \
                --no-cache-dir \
                --index-url https://download.pytorch.org/whl/cpu \
                --extra-index-url https://pypi.org/simple

COPY ./app.py /app/
COPY ./src/ /app/src/

ENTRYPOINT [ "python3", "/app/app.py" ]