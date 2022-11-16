FROM python:3.8-slim
ENV PORT=8000
COPY requirements.txt /
RUN pip install -r requirements.txt
EXPOSE 8000
COPY ./app /app
COPY New_Model.h5 /
COPY model_MOBILNETv2.h5 /
COPY clases_tipologia.txt /
COPY clases.txt /
RUN pip3 install opencv-python-headless

ENTRYPOINT uvicorn app.main:app --host 0.0.0.0 --port $PORT 


