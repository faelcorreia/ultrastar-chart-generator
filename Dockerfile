FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep MFA models outside the build user's home so they are available when the
# container runs as the host user (docker.sh passes -u UID:GID).
ENV MFA_ROOT_DIR=/opt/mfa

RUN conda create -n audio-processor python=3.10 -y && \
    conda install -n audio-processor -c conda-forge montreal-forced-aligner -y && \
    conda run -n audio-processor mfa models download acoustic portuguese_mfa && \
    conda run -n audio-processor mfa models download dictionary portuguese_brazil_mfa && \
    conda run -n audio-processor mfa models download g2p portuguese_brazil_mfa && \
    conda run -n audio-processor mfa models download acoustic english_mfa && \
    conda run -n audio-processor mfa models download dictionary english_us_mfa && \
    conda run -n audio-processor mfa models download g2p english_us_mfa && \
    conda run -n audio-processor mfa models download acoustic spanish_mfa && \
    conda run -n audio-processor mfa models download dictionary spanish_latin_america_mfa && \
    conda run -n audio-processor mfa models download g2p spanish_latin_america_mfa && \
    chmod -R a+rwX /opt/mfa && \
    conda clean -afy

COPY ./requirements.txt /app/

RUN conda run -n audio-processor pip install -r requirements.txt \
                --no-cache-dir \
                --index-url https://download.pytorch.org/whl/cpu \
                --extra-index-url https://pypi.org/simple

COPY ./app.py /app/
COPY ./src/ /app/src/

ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "audio-processor", "python3", "/app/app.py" ]
