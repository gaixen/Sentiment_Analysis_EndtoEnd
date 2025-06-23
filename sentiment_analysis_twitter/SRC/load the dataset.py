from dotenv import load_dotenv
import os
import tweepy


load_dotenv()
TWITTER_CLIENT_ID = os.getenv("TWITTER_CLIENT_ID")
TWITTER_CLIENT_SECRET = os.getenv("TWITTER_CLIENT_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# auth = tweepy.OAuth2AppHandler(
#     TWITTER_CLIENT_ID, TWITTER_CLIENT_SECRET
# )

client = tweepy.Client(bearer_token = TWITTER_BEARER_TOKEN)

# api = tweepy.API(auth)

# print(api.rate_limit_status())

query = "#IranIsraelConflict -is:retweet lang:en" 
tweets = client.search_recent_tweets(query=query, max_results=10, tweet_fields=["created_at", "lang", "text"])

#collect tweet texts
tweet_texts = [tweet.text for tweet in tweets.data] if tweets.data else []

with open(r'C:\Users\sudip\LLM_Fine_Tuning\outputs\tweets.txt', "w", encoding="utf-8") as f:    
    for tweet in tweet_texts:
        f.write(tweet + "\n")
        
#print sample tweets
for i, text in enumerate(tweet_texts[:5], 1):
    print(f"{i}. {text}")
