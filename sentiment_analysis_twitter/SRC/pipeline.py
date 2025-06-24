from transformers import pipeline
import pandas as pd

pipe = pipeline("sentiment-analysis")

df = pd.read_csv(r'C:\Users\sudip\LLM_Fine_Tuning\outputs\extracted_tweets.csv')

df[['label', 'score']] = df['text'].apply(lambda x: pd.Series(pipe(x)[0]))

output_file_path = r'C:\Users\sudip\LLM_Fine_Tuning\outputs\extracted_tweets.csv'
df.to_csv(output_file_path, index=False, encoding='utf-8')
