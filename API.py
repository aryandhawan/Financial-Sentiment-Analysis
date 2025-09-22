import pandas as pd
import requests
import time
bearer_token='AAAAAAAAAAAAAAAAAAAAALT14AEAAAAAKvupa%2Fe337uGek%2BYZhbpmHtQYTM%3DZo948kPYKGCEYJrEhkeoOsxeJlbEfqY6JiNsauK5kWUrbikjYw'


API_key='ukgADHC0kmH2Dl2LFJcm8loKy'
API_key_secret='GKpdzGEKMzWjxzI905VC16Sx5pEyRLdtRd19lbC6WQhbi6c7Fg'

url='https://api.x.com/2/tweets/search/recent'

query_string="from:elonmusk (Tesla OR TSLA OR Cybertruck)"

params = {
    'query': query_string,
    'max_results': 10,
    'tweet.fields': 'created_at,text,public_metrics'
}

headers = {
    "Authorization": f"Bearer {bearer_token}"
}

# making the API request call

response=requests.get(url,headers=headers)
if response.status_code==200:
    print("Sucess found the data")
    if 'data' in response.json() and response.json['data']:
        tweets_data=response.json()['data']
        processed_tweets=[]
        for tweet in tweets_data:
            processed_tweets.append({
                "created_at": tweet['created_at'],
                "text": tweet['text'],
                "retweet_count": tweet['public_metrics']['retweet_count'],
                "like_count": tweet['public_metrics']['like_count'],
            })

            # now its time to convert it into a dataframe
            df=pd.DataFrame(processed_tweets)
            print("Sucessfully converted to a dataframe")
            print(df.head())

elif response.status_code == 429:
    print("\nERROR: 429 Too Many Requests. Rate limit exceeded.")
    print("The API is telling us to slow down.")
    
    # A professional way to handle this is to check the 'x-rate-limit-reset' header
    # This tells us when we can make a request again (as a Unix timestamp)
    reset_time = int(response.headers.get('x-rate-limit-reset', 0))
    current_time = int(time.time())
    
    wait_time = max(0, reset_time - current_time)
    
    if wait_time > 0:
        print(f"Waiting for {wait_time} seconds before the rate limit resets...")
        # time.sleep(wait_time) # In a real script, you would wait
    else:
        print("Rate limit should reset soon. Try again in a minute.")

else:
    print(f"\nAn unexpected error occurred: {response.status_code}")
    print(response.text)

