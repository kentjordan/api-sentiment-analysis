FROM python:latest

COPY . /api/

WORKDIR /api/

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install fastapi pandas numpy nltk tweet-preprocessor setuptools wheel spacy
RUN python -m spacy download en_core_web_sm

WORKDIR /api/src/

CMD ["fastapi", "run", "api.py"]