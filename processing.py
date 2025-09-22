import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. Load the Raw Data ---
print("Loading raw tweet data...")
# Using low_memory=False is good practice for potentially mixed-type columns.
df = pd.read_csv('all_musk_posts.csv', low_memory=False)
print(f"✓ Loaded {len(df)} tweets.")

# --- 2. Initial Cleaning (Your Existing Code) ---
print("Performing initial cleaning...")
# Drop columns that are not needed for this analysis.
columns_to_drop = [
    'bookmarkCount', 'isReply', 'inReplyToId', 'conversationId',
    'inReplyToUserId', 'inReplyToUsername', 'isPinned', 'isRetweet',
    'isQuote', 'isConversationControlled', 'possiblySensitive',
    'quoteId', 'quote', 'retweet', 'id', 'url', 'twitterUrl'
]
df_new = df.drop(columns=columns_to_drop)

# As we discovered, 'viewCount' has too many historical NULLs and is a noisy signal.
df_new.drop('viewCount', axis=1, inplace=True)

# Drop any remaining rows with nulls in key engagement metrics.
df_new.dropna(subset=['likeCount', 'retweetCount', 'replyCount', 'quoteCount'], inplace=True)
print("✓ Initial cleaning complete.")

# --- 3. Feature Engineering: Sentiment Analysis ---
print("Starting sentiment analysis...")

# Initialize the VADER analyzer once.
analyzer = SentimentIntensityAnalyzer()


def get_sentiment_score(sentence):
    """
    Analyzes a text string and returns its VADER compound sentiment score.
    """
    # Ensure the input is a string, handle potential float/NaN values.
    if not isinstance(sentence, str):
        return 0.0  # Return a neutral score for non-text data

    score = analyzer.polarity_scores(sentence)
    return score['compound']


# Apply the function to the 'fullText' column to create the new feature.
# This is the core NLP step of our feature engineering.
df_new['sentiment_score'] = df_new['fullText'].apply(get_sentiment_score)
print("✓ Sentiment analysis complete. 'sentiment_score' column created.")

# --- 4. Save the Processed Data ---
# This is the final output of our script: a clean, enriched dataset
# that is ready for the next phase (merging with stock data).
OUTPUT_PATH = 'processed_tweets.csv'
df_new.to_csv(OUTPUT_PATH, index=False)
print(f"✓ Fully processed data saved to '{OUTPUT_PATH}'")

# --- 5. Final Inspection ---
print("\nHere is a sample of your final processed data:")
print(df_new[['createdAt', 'fullText', 'sentiment_score', 'likeCount']].head())
