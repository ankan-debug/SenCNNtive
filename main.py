# main.py

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import praw

# This MUST be the same value as in train_model.py
MAX_LEN = 64

# --- 1. Function to Extract Text from Reddit ---
def get_text_from_reddit(url):
    """
    Extracts the title and body text from a Reddit submission URL using PRAW.
    """
    try:
        reddit = praw.Reddit(
            client_id="HKYQRKlb_oPCHnhjIS7Khg",       
            client_secret="xPctA_Tbqpdp-WXrFiJZgIaxm2Mm6Q",  
            user_agent="sentiment tool v1 by u/Ankan_Saha" 
        )
        
        print(f"Fetching post from: {url}")
        submission = reddit.submission(url=url)
        # Combine the title and the post's body text (selftext)
        full_text = submission.title + " " + submission.selftext
        print("Fetch successful.")
        return full_text
    
    except Exception as e:
        print(f"Error fetching from Reddit: {e}")
        return f"Error: Could not fetch post. Check URL or API credentials.\nDetails: {e}"

# --- 2. Function to Predict Sentiment ---
def predict_sentiment(text, model, tokenizer):
    """
    Takes raw text and predicts sentiment using the loaded Keras model.
    """
    try:
        # Prepare the text for the model
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Get the prediction (a list of 3 probabilities)
        prediction = model.predict(padded_sequence, verbose=0)[0]
        
        # Find the index with the highest probability
        class_index = np.argmax(prediction)
        confidence = prediction[class_index]
        
        # Label mapping for TweetEval: 0: negative, 1: neutral, 2: positive
        if class_index == 0:
            sentiment = "NEGATIVE 😠"
        elif class_index == 1:
            sentiment = "NEUTRAL 😐"
        else: # class_index == 2
            sentiment = "POSITIVE 😊"
            
        return f"{sentiment}\n(Confidence: {confidence*100:.2f}%)"
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Error: Could not analyze text.\nDetails: {e}"