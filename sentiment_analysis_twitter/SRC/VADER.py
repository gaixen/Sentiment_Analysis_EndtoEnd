import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

df = pd.read_csv(r'C:\Users\sudip\LLM_Fine_Tuning\outputs\extracted_tweets.csv')
df['id'] = range(1 , len(df)+1)

df[['neg' , 'neu' , 'pos' , 'compound']] = df['text'].apply(lambda x: pd.Series(sia.polarity_scores(x)))

df['sentiment'] = df['compound'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))

output_file_path = r'C:\Users\sudip\LLM_Fine_Tuning\outputs\extracted_tweets.csv'
df.to_csv(output_file_path, index=False, encoding='utf-8')
print(f"Sentiment analysis completed. Results saved to {output_file_path}")
