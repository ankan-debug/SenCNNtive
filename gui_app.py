# gui_app.py

import tkinter as tk
from tkinter import scrolledtext
import threading
import os
from main import get_text_from_reddit, predict_sentiment # Imports from main.py
from tensorflow.keras.models import load_model
import pickle

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SenCNNtive")
        self.root.geometry("500x400")

        # --- Create and place GUI widgets ---
        self.url_label = tk.Label(root, text="Enter Social Media Post URL:", font=("Helvetica", 12))
        self.url_label.pack(pady=(10,5))

        self.url_entry = tk.Entry(root, width=60, font=("Helvetica", 10))
        self.url_entry.pack(pady=5, padx=10)

        self.analyze_button = tk.Button(root, text="Analyze Sentiment", command=self.start_analysis, font=("Helvetica", 11, "bold"))
        self.analyze_button.pack(pady=10)

        self.result_label = tk.Label(root, text="Results:", font=("Helvetica", 12))
        self.result_label.pack(pady=(10,5))

        self.result_text = scrolledtext.ScrolledText(root, width=55, height=10, wrap=tk.WORD, font=("Helvetica", 10))
        self.result_text.pack(pady=5, padx=10)
        self.result_text.configure(state='disabled')

        self.model = None
        self.tokenizer = None
        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        self.update_result("Loading Keras model and tokenizer...")
        try:
            self.model = load_model('sentiment_cnn.h5')
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            self.update_result("✅ Model and tokenizer loaded successfully. Developer @ankan_saha")
        except IOError:
            self.update_result("❌ ERROR: Model/tokenizer files not found.\nPlease run 'python train_model.py' first.")
            self.analyze_button.configure(state='disabled')

    def start_analysis(self):
        self.analyze_button.configure(state='disabled')
        self.result_text.configure(state='normal')
        self.result_text.delete('1.0', tk.END)
        self.result_text.configure(state='disabled')
        self.update_result("Starting analysis...")
        
        thread = threading.Thread(target=self.perform_analysis)
        thread.daemon = True
        thread.start()
        
    def perform_analysis(self):
        post_url = self.url_entry.get()
        if not post_url:
            self.update_result("Please enter a URL.", re_enable_button=True)
            return

        extracted_text = None
        if 'reddit.com' in post_url:
            self.update_result("Fetching content from Reddit...")
            extracted_text = get_text_from_reddit(post_url)
        else:
            self.update_result("Sorry, this tool currently only supports Reddit links.", re_enable_button=True)
            return

        if extracted_text:
            self.update_result("✅ Text extracted. Predicting sentiment...")
            sentiment = predict_sentiment(extracted_text, self.model, self.tokenizer)
            self.update_result(f"--- Analysis Complete ---\nPredicted Sentiment: {sentiment} \nMade with ❤️ by Ankan", re_enable_button=True)
        else:
            self.update_result("Failed to extract text from the URL.", re_enable_button=True)

    def update_result(self, message, re_enable_button=False):
        def task():
            self.result_text.configure(state='normal')
            self.result_text.insert(tk.END, message + "\n")
            self.result_text.configure(state='disabled')
            self.result_text.see(tk.END)
            
            if re_enable_button:
                self.analyze_button.configure(state='normal')

        self.root.after(0, task)

# --- Main Execution Block ---
if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()