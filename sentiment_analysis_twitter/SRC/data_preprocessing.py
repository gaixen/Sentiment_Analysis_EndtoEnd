import re
import pandas as pd
import string , time
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
# convert tweets .txt to dataframe

def convert_tweets_to_dataframe(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tweets = f.readlines()
    tweets = [tweet.strip() for tweet in tweets if tweet.strip()]

    df = pd.DataFrame(tweets, columns=['text'])
    return df

df = convert_tweets_to_dataframe(r'C:\Users\sudip\LLM_Fine_Tuning\outputs\tweets.txt')

df.to_csv(r'C:\Users\sudip\LLM_Fine_Tuning\outputs\extracted_tweets.csv', index=False, encoding='utf-8')

chat_words = {
    'AFAIK':'As Far As I Know',
    'AFK':'Away From Keyboard',
    'ASAP':'As Soon As Possible',
    'FYI': 'For Your Information',
    'ASAP': 'As Soon As Possible',
    'BRB': 'Be Right Back',
    'BTW': 'By The Way',
    'OMG': 'Oh My God',
    'IMO': 'In My Opinion',
    'LOL': 'Laugh Out Loud',
    'TTYL': 'Talk To You Later',
    'GTG': 'Got To Go',
    'TTYT': 'Talk To You Tomorrow',
    'IDK': 'I do not Know',
    'TMI': 'Too Much Information',
    'IMHO': 'In My Humble Opinion',
    'ICYMI': 'In Case You Missed It',
    'AFAIK': 'As Far As I Know',
    'BTW': 'By The Way',
    'FAQ': 'Frequently Asked Questions',
    'TGIF': 'Thank God It is Friday',
    'FYA': 'For Your Action',
    'ICYMI': 'In Case You Missed It',
}

def replace_chat_words(text):
    for word, replacement in chat_words.items():
        text = text.replace(word, replacement)
    return text

df['text'] = df['text'].apply(replace_chat_words)

df['text'] = df['text'].apply(lambda x: str.lower(x))

def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

df['text'] = df['text'].apply(remove_html_tags)

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

df['text'] = df['text'].apply(remove_url)

exclude = string.punctuation
def remove_punc(text):
    for char in exclude:
        text = text.replace(char , '')
    return text

df['text'] = df['text'].apply(remove_punc)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F" 
                        u"\U0001F300-\U0001F5FF"  
                        u"\U0001F680-\U0001F6FF"  
                        u"\U0001F1E0-\U0001F1FF" 
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

df['text'] = df['text'].apply(remove_emoji)

# def text_correction(text):
#     blob = TextBlob(text)
#     corrected_text = str(blob.correct())
#     return corrected_text

# df['text'] = df['text'].apply(text_correction)

def remove_stopwords(text):
    new_text = []
    for word in text.split():
        if word in stopwords.words('english'):
            new_text.append('')
        else:
            new_text.append(word)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)

df['text'] = df['text'].apply(remove_stopwords)

# Save the cleaned DataFrame to a new CSV file
df.to_csv(r'C:\Users\sudip\LLM_Fine_Tuning\outputs\cleaned_tweets.csv', index=False, encoding='utf-8')
