import numpy as np
import pickle
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# --- New Imports for Evaluation ---
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# --- Parameters ---
VOCAB_SIZE = 10000    # Number of words to keep in the vocabulary
MAX_LEN = 64          # Max length of sequences
EMBEDDING_DIM = 128   # Dimension of word embeddings
# Define class names for reports
CLASS_NAMES = ['Negative', 'Neutral', 'Positive']

print("Loading 'tweet_eval' sentiment dataset...")
# Load the dataset from Hugging Face
dataset = load_dataset("tweet_eval", "sentiment")
train_data = dataset['train']
test_data = dataset['test'] # <-- Load the test set for evaluation

# Extract train text and labels
texts = train_data['text']
labels = np.array(train_data['label'])

print(f"Loaded {len(texts)} training examples.")

# --- 1. Tokenize Text ---
print("Fitting tokenizer on text data...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<UNK>")
tokenizer.fit_on_texts(texts)

# Convert text to sequences of integers
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to be the same length
X_train = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
y_train = labels

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")

# --- Process Test Data ---
print("Processing test data for evaluation...")
test_texts = test_data['text']
y_test = np.array(test_data['label'])
test_sequences = tokenizer.texts_to_sequences(test_texts)
X_test = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")


# --- 2. Define the CNN Model ---
print("Building the Keras CNN model...")
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, 
              output_dim=EMBEDDING_DIM, 
              input_length=MAX_LEN),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax') # 3 units for 3 classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Use this for integer labels (0, 1, 2)
    metrics=['accuracy']
)

model.summary()

# --- 3. Train the Model ---
print("\nStarting model training... (This may take a while)")
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,  
    verbose=1
)

# --- 4. Save the Model and Tokenizer ---
print("Training complete. Saving model and tokenizer...")
model.save('sencnntive_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model and tokenizer saved successfully.")

# =================================================================
# ⬇️⬇️ MODIFIED EVALUATION SECTION ⬇️⬇️
# =================================================================

print("\n--- Model Evaluation ---")

# --- 5. Generate "Model Fit" Plots ---
print("Generating 'Model Fit' plots...")

# Plot 1: Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Fit: Training vs. Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('model_accuracy.png') # Save plot as an image
print("Saved 'model_accuracy.png'")

# Plot 2: Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Fit: Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('model_loss.png') # Save plot as an image
print("Saved 'model_loss.png'")


# --- 6. Generate Classification Report & Confusion Matrix ---
print("\nEvaluating model on test data...")

# Get predictions on the test set
predictions_prob = model.predict(X_test, verbose=0)
# Convert probabilities to class labels (0, 1, or 2)
y_pred = np.argmax(predictions_prob, axis=1)

# --- Classification Report ---
print("\n--- Classification Report ---")
report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
print(report)
# Save report to a text file
with open('classification_report.txt', 'w') as f:
    f.write(report)
print("Saved 'classification_report.txt'")


# --- Confusion Matrix ---
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
# Format the matrix as a string to save
cm_header = "       (Predicted)\n       Neg  Neu  Pos\n---------------------\n"
cm_body = ""
for i, row in enumerate(cm):
    cm_body += f"(Actual {CLASS_NAMES[i][:3]}) {row[0]:<4} {row[1]:<4} {row[2]:<4}\n"
    
print(cm_header + cm_body)
# Save matrix to a text file
with open('confusion_matrix.txt', 'w') as f:
    f.write(cm_header + cm_body)
print("Saved 'confusion_matrix.txt'")


print("\nAll done!")