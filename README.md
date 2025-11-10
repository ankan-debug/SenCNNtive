
# SenCNNtive: A Keras-Powered Sentiment Analysis GUI 🧠

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Keras%20%2F%20TensorFlow-2.x-orange.svg" alt="Keras/TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<p align="center">
  <i>A desktop application that analyzes the sentiment of Reddit posts using a Convolutional Neural Network (CNN) and provides a full evaluation report.</i>
</p>

SenCNNtive is a Python-based tool that uses a deep learning model to classify the sentiment of Reddit posts as **Positive**, **Negative**, or **Neutral**. The app features a custom "tech-themed" Tkinter GUI, pulls data directly from Reddit, and includes a full model evaluation dashboard.

## 🚀 Key Features

* **Deep Learning Core**: Utilizes a CNN built with Keras/TensorFlow to capture contextual features from text.
* **Live Reddit Data**: Integrates with the Reddit API (PRAW) to fetch any post's title and body text for analysis.
* **Modern Dataset**: Trained on the Hugging Face `datasets` library (`TweetEval` task) for high performance on modern social media language.
* **Full Evaluation Suite**: The app can display a complete model performance report, including:
    * Model Fit (Accuracy vs. Validation Accuracy)
    * Model Loss (Loss vs. Validation Loss)
    * Classification Report (Precision, Recall, F1-Score)
    * Confusion Matrix
* **Custom GUI**: A user-friendly graphical interface built with Python's native Tkinter library, styled with a dark, "techie" theme.

## 🛠️ Technology Stack

* **Machine Learning**: Keras / TensorFlow, Scikit-learn
* **Data Handling**: NumPy, Hugging Face `datasets`
* **API Integration**: PRAW (Python Reddit API Wrapper)
* **GUI**: Tkinter, Pillow (PIL)
* **Plotting**: Matplotlib

---

## ⚙️ Getting Started: Installation & Usage

Follow these steps to get the project running on your local machine.

### Prerequisites

* [Python (3.10 or newer)](https://www.python.org/downloads/)
* [Git](https://git-scm.com/downloads)

### Step 1: Clone the Repository

First, open your terminal (CMD, PowerShell, etc.) and clone this repository.

```bash
git clone [https://github.com/ankan-debug/SenCNNtive.git](https://github.com/ankan-debug/SenCNNtive.git)
cd SenCNNtive
````

### Step 2: Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the environment
python -m venv venv

# Activate the environment (Windows)
.\venv\Scripts\activate

# Activate the environment (macOS/Linux)
# source venv/bin/activate
```

### Step 3: Install Dependencies

With your virtual environment active, install all the required libraries.

```bash
pip install -r requirements.txt
```

### Step 4: Add Your Reddit API Keys (CRITICAL)

This application will **not** work without your personal API keys.

1.  Go to [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) and create a new "script" application.

2.  Open the `main.py` file in a code editor.

3.  Find the `get_text_from_reddit` function.

4.  Replace the placeholder values with your keys:

    ```python
    # main.py

    ...
    reddit = praw.Reddit(
        client_id="YOUR_CLIENT_ID_HERE",        # <-- PASTE YOUR ID
        client_secret="YOUR_CLIENT_SECRET_HERE",  # <-- PASTE YOUR SECRET
        user_agent="sentiment tool v1 by u/YOUR_USERNAME_HERE" # <-- ADD YOUR USERNAME
    )
    ...
    ```

**Warning:** Never commit your secret keys to GitHub.

### Step 5: Train the Model (One-Time Step)

Before you can run the app, you must run the training script. This will:

1.  Download the `TweetEval` dataset.
2.  Train the CNN model.
3.  Save the `sencnntive_model.h5` and `tokenizer.pickle` files.
4.  Generate all evaluation files (`.png` and `.txt`).

This process may take several minutes.

```bash
python train_model.py
```

### Step 6: Run the Application\!

Once training is complete, you can launch the GUI.

```bash
python gui_app.py
```

## 🖥️ How to Use the App

1.  Find a Reddit post you want to analyze.
2.  Copy the full URL of the post.
3.  Paste the URL into the text box in the application.
4.  Click the **"Analyze Sentiment"** button.
5.  (Optional) Click the **"Show Model Evaluation"** button to open a new window displaying the model's performance reports.

-----

## ❤️ Author

Made With ❤️ By **Ankan**

```
```
