FROM registry.access.redhat.com/ubi8/python-311:1-43

WORKDIR /app

COPY ../../requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./components/VideoToFrame/VideoToFrame.py /app/VideoToFrame.py
COPY ../../wf_base_component.py /app/wf_base_component.py
COPY ../../wf_component_types.py /app/wf_component_types.py
COPY ../../wf_data_types.py /app/wf_data_types.py

RUN pip install opencv-python-headless
