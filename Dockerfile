FROM python:3.9

ADD stm.py .

RUN pip install streamlit


CMD ["streamlit","run","./stm.py"]