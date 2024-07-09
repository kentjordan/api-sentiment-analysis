from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils import make_input

import pandas as pd

import torch
from model import Classifier

class SentimentBody(BaseModel):
    text: str

model_path = torch.load("../models/release/model.pt")
dictionary = pd.read_csv("../data/release/dictionary.csv")

model = Classifier(in_features=3)
model.load_state_dict(model_path)

app = FastAPI()

origins =  [
    "http://localhost:3000",
    "https://sentiment-analysis-kjordan.vercel.app"
]

app.add_middleware(CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/")
def root():
    return "Sentiment Analysis API made by Kent Jordan with 💖."

@app.post("/analyze")
def analyze(body: SentimentBody):

    vocabulary = dictionary['vocabulary'].to_list()
    pos_freqs = dictionary['pos_freq'].to_list()
    neg_freqs = dictionary['neg_freq'].to_list()
    
    x = make_input(body.text, scaling="sqrt", pos_freqs=pos_freqs, neg_freqs=neg_freqs, vocabulary=vocabulary)
    yhat =  model(x)

    return {
        "probability": yhat.item(),
        "sentiment": 1 if yhat.item() > 0.5 else 0
    }