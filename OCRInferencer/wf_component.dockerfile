FROM docker.io/kkwterence/mmocr:ver4.1
# FROM registry.access.redhat.com/ubi8/python-311:1-43

WORKDIR /app

COPY ../../requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./components/OCRInferencer/assets /app/assets
COPY ./components/OCRInferencer/src /app/src

COPY ./components/OCRInferencer/OCRInferencer.py /app/OCRInferencer.py
COPY ../../wf_base_component.py /app/wf_base_component.py
COPY ../../wf_component_types.py /app/wf_component_types.py
COPY ../../wf_data_types.py /app/wf_data_types.py

# COPY ./components/OCRInferencer/__init__.py /app/__init__.py





# FROM registry.access.redhat.com/ubi8/python-311:1-43

# WORKDIR /app

# COPY ../../requirements.txt /app/requirements.txt
# RUN pip install --upgrade pip && pip install -r requirements.txt

# COPY ./components/OCRInferencer/OCRInferencer.py /app/OCRInferencer.py
# COPY ../../wf_base_component.py /app/wf_base_component.py
# COPY ../../wf_component_types.py /app/wf_component_types.py
# COPY ../../wf_data_types.py /app/wf_data_types.py