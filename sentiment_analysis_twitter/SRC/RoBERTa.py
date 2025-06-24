from transformers import AutoTokenizer
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
import torch
import pandas as pd

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_Roberta(text):
    encoded_input = tokenizer(text, return_tensors='pt',truncation = True,max_length = 512)
    with torch.no_grad():
        output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    dict_Roberta = {
        'Roberta_neg': scores[0],
        'Roberta_neu': scores[1],
        'Roberta_pos': scores[2]
    }
    return dict_Roberta

df = pd.read_csv(r'C:\Users\sudip\LLM_Fine_Tuning\outputs\extracted_tweets.csv')

df[['Roberta_neg' , 'Roberta_neu' , 'Roberta_pos']] = df['text'].apply(lambda x: pd.Series(polarity_scores_Roberta(x)))

df.to_csv(r'C:\Users\sudip\LLM_Fine_Tuning\outputs\extracted_tweets.csv', index=False, encoding='utf-8')
