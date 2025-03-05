FROM dntextspotter_ubi8_devel:v1-weaveforge

WORKDIR /app

USER 1001

COPY ./assets/ /app/assets/
COPY ./src/ /app/src/
COPY ./OCRInferencer.py /app/OCRInferencer.py