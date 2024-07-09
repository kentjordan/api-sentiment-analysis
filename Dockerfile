FROM python:latest

COPY . /api/

WORKDIR /api/

RUN pip install fastapi pandas numpy torch nltk preprocessor setuptools wheel spacy
RUN python -m spacy download en_core_web_sm

CMD ["fastapi", "run", "api.py"]