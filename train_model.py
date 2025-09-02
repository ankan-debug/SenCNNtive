# train_model.py (Final Corrected Version)

from datasets import load_dataset, concatenate_datasets # <-- Note the new import here
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import pickle

# --- Parameters ---
VOCAB_SIZE = 15000
MAX_LEN = 64
EMBEDDING_DIM = 128

# --- 1. Load Dataset from Hugging Face ---
print("Step 1: Loading TweetEval dataset from Hugging Face...")
try:
    dataset = load_dataset("tweet_eval", "sentiment")
except Exception as e:
    print(f"Failed to load dataset. Please check your internet connection. Error: {e}")
    exit()

# --- THIS IS THE CORRECTED SECTION ---
# Combine the train and validation splits into a single dataset object
combined_dataset = concatenate_datasets([dataset['train'], dataset['validation']])

# Now extract the texts and labels from the new combined dataset and the test set
train_texts = combined_dataset['text']
train_labels = combined_dataset['label']
test_texts = dataset['test']['text']
test_labels = dataset['test']['label']
# --- END OF CORRECTION ---


# --- 2. Tokenize and Pad Text ---
print("Step 2: Tokenizing and padding sequences...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=MAX_LEN, padding='post', truncating='post')

y_train = np.array(train_labels)
y_test = np.array(test_labels)

# --- 3. Build the CNN Model ---
print("Step 3: Building the Keras CNN model for multi-class classification...")
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 4. Train the Model ---
print("\nStep 4: Training the model...")
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=2)

# --- 5. Evaluate and Save ---
print("\nStep 5: Evaluating and saving the model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")

print("Saving model to 'sentiment_cnn.h_5'...")
model.save('sentiment_cnn.h5')

print("Saving tokenizer to 'tokenizer.pickle'...")
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\n✅ Training complete! Your model and tokenizer are ready.")