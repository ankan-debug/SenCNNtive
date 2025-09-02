# SenCNNtive: A Keras-Powered Sentiment Analysis GUI 🧠

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Keras-3-red" alt="Keras Version">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

SenCNNtive is a desktop application that analyzes the sentiment of social media posts using a **Convolutional Neural Network (CNN)**. The model is built with Keras/TensorFlow and trained on the modern **TweetEval** dataset to understand the nuances of online language, classifying posts as **Positive**, **Negative**, or **Neutral**.



---

## Key Features

-   **Deep Learning Core**: Utilizes a CNN to capture contextual features from text for accurate sentiment classification.
-   **Modern Dataset**: Trained on the Hugging Face `TweetEval` dataset, ensuring better performance on current social media language.
-   **Multi-Class Analysis**: Classifies text into three distinct categories: Positive, Negative, and Neutral, with a confidence score.
-   **Simple GUI**: A user-friendly graphical interface built with Python's native Tkinter library for ease of use.

---

## Technology Stack

-   **Backend**: Python
-   **Deep Learning**: Keras 3 with a TensorFlow backend
-   **Data Handling**: Hugging Face `datasets`, NumPy
-   **API Integration**: Python Reddit API Wrapper (PRAW)
-   **GUI**: Tkinter

---

## ⚠️ Current Status: Reddit Only

In its current version, this tool is configured to analyze posts exclusively from **Reddit**. The content extractor is built using the official Reddit API (PRAW). Support for other platforms is planned for future updates.

---

## 🚀 Future Roadmap

This project is a foundation for a more comprehensive social media analysis tool. Future development goals include:

-   [ ] **Adding Twitter/X Support:** Integrating the X API to analyze tweets.
-   [ ] **Model Improvement:** Experimenting with more advanced architectures like LSTMs or Transformers (e.g., DistilBERT).
-   [ ] **Executable Packaging:** Bundling the application into a standalone `.exe` file using PyInstaller for easy distribution.

---

## 🛠️ Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

-   Python 3.10 or newer
-   Git

### Installation & Usage

1.  **Clone the Repository**
    Open your terminal and run the following command to clone the project.
    ```bash
    git clone https://github.com/ankan-debug/SenCNNtive.git
    ```

2.  **Navigate to the Project Directory**
    ```bash
    cd SenCNNtive
    ```

3.  **Set Up the Virtual Environment**
    Create and activate a Python virtual environment.
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

4.  **Install Dependencies**
    Install all the required libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure API Keys**
    Before running the application, you must add your own Reddit API credentials.
    -   Open the **`main.py`** file.
    -   Find the `get_text_from_reddit` function.
    -   Replace the placeholder values for `client_id` and `client_secret` with your own keys.

6.  **Train the Model (One-Time Step)**
    You must train the model once to generate the necessary `.h5` and `.pickle` files. This is a long, resource-intensive process.
    ```bash
    python train_model.py
    ```

7.  **Run the Application!**
    Once training is complete, launch the GUI application.
    ```bash
    python gui_app.py
    ```
    The application window will now open, ready for you to analyze Reddit posts!
