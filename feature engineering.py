import pandas as pd
import numpy as np

print('Starting feature engineering')

# Load the data first
try:
    tweets_df = pd.read_csv('processed_tweets.csv')
    stock_df = pd.read_csv('stock data.csv')
    print('All files uploaded')
except FileNotFoundError as e:
    print("------Error-----")
    print(f"Could not find the file: {e}. Please check the filename and path.")
    exit()

# Prepare the timestamps for processing
print("Standardizing timestamps...")
tweets_df['timestamp'] = pd.to_datetime(tweets_df['createdAt'])
tweets_df['timestamp'] = tweets_df['timestamp'].dt.tz_convert('UTC')

stock_df['timestamp'] = pd.to_datetime(stock_df['Datetime'], utc=True)

# Set both the times as the index of the dataset
tweets_df.set_index('timestamp', inplace=True)
stock_df.set_index('timestamp', inplace=True)
print('Standardisation is done')

print("Merging tweet data with stock data...")
merged_df = pd.merge_asof(
    left=tweets_df.sort_index(),
    right=stock_df[['Close']].sort_index(),
    left_index=True,
    right_index=True,
    direction='backward'
)
print("Merge completed")
# doing feature engineering
print('adding new features or Feature engineering')
merged_df['sentiment_impact']=merged_df['sentiment_score']*merged_df['likeCount']+1
# time based feature extraction
merged_df['hour_of_day']=merged_df.index.hour
merged_df['day_of_week']=merged_df.index.dayofweek

# Lag features (addtional)

merged_df['prior_close_1h']=merged_df['Close'].shift(1)
merged_df['prior_1h_change']=(merged_df['Close']-merged_df['prior_close_1h'])/merged_df['prior_close_1h']*100
print('New features have been added!')
merged_df['future_close_24h'] = merged_df['Close'].shift(-24)

# --- FIX 2: Calculate percentage change correctly ---
merged_df['price_change_24h'] = (merged_df['future_close_24h'] - merged_df['Close']) / merged_df['Close'] * 100

def create_label(change):
    if change > 0.5:
        return 'Up'
    # --- FIX 3: Correct the logic for 'Down' ---
    elif change < -0.5:
        return "Down"
    else:
        return 'Neutral'

merged_df['Trend'] = merged_df['price_change_24h'].apply(create_label)
print("Target variable 'Trend' is created successfully")
# Drop rows where the future price could not be calculated
final_df = merged_df.dropna()

model_ready_df = final_df[[
    'fullText',
    'sentiment_score',
    'likeCount',
    'retweetCount',
    'sentiment_impact',
    'hour_of_day',
    'day_of_week',
    'prior_1h_change',
    'Trend'
]]

OUTPUT_PATH = 'final_training_dataset.csv'
model_ready_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Success! Final model-ready dataset saved to '{OUTPUT_PATH}'")
print(f"Dataset shape: {model_ready_df.shape}")
print("\nSample of the final data:")
print(model_ready_df.head())
print('Done')

