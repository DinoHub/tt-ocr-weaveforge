FROM registry.access.redhat.com/ubi9/python-311@sha256:fc669a67a0ef9016c3376b2851050580b3519affd5ec645d629fd52d2a8b8e4a

WORKDIR /app

USER 1001

COPY ./VideoToFrame.py /app/VideoToFrame.py
COPY ./weaveforge-0.1.0-py3-none-any.whl /app/weaveforge-0.1.0-py3-none-any.whl
RUN pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip pip install /app/weaveforge-0.1.0-py3-none-any.whl
RUN pip install opencv-python-headless minio
