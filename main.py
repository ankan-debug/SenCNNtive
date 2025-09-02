# main.py

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import praw

# --- Parameters ---
# This MUST be the same value used in your train_model.py script
MAX_LEN = 64

# --- 1. Function to Extract Text from Reddit ---
def get_text_from_reddit(url):
    """
    Extracts the title and body text from a Reddit submission URL using PRAW.
    """
    try:
        # IMPORTANT: You must get these credentials from Reddit
        # Go to: https://www.reddit.com/prefs/apps
        reddit = praw.Reddit(
            client_id="HKYQRKlb_oPCHnhjIS7Khg",
            client_secret="xPctA_Tbqpdp-WXrFiJZgIaxm2Mm6Q",
            user_agent="sentiment analysis tool v1 by u/YourUsername" # Change YourUsername
        )
        submission = reddit.submission(url=url)
        full_text = submission.title + " " + submission.selftext
        return full_text
    except Exception as e:
        print(f"Error fetching from Reddit: {e}")
        return None

# --- 2. Function to Predict Sentiment ---
def predict_sentiment(text, model, tokenizer):
    """
    Takes raw text and predicts sentiment using the multi-class Keras model.
    """
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    prediction = model.predict(padded_sequence, verbose=0)[0]
    
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]
    
    # Label mapping for TweetEval: 0: negative, 1: neutral, 2: positive
    if class_index == 0:
        return f"NEGATIVE (Confidence: {confidence*100:.2f}%)"
    elif class_index == 1:
        return f"NEUTRAL (Confidence: {confidence*100:.2f}%)"
    else: # class_index == 2
        return f"POSITIVE (Confidence: {confidence*100:.2f}%)"