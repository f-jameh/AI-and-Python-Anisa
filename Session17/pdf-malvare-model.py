
# In the name of God
import os
import pandas as pd

dataset_dir = "/home/farhad/Desktop/data/datasets/smal-cic-pdf-malware"  # Change this to your dataset folder path
labels = []

# Function to check if a file is a valid PDF (based on header signature)
def is_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
            return header == b"%PDF"
    except:
        return False  # If any error occurs, assume it's not a PDF

# Process benign files
benign_path = os.path.join(dataset_dir, "Benign")
for folder in os.listdir(benign_path):
    folder_path = os.path.join(benign_path, folder)
    if os.path.isdir(folder_path):  # Ensure it's a folder
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and is_pdf(file_path):  # Check if it's really a PDF
                labels.append((file_path, 0))  # 0 = Benign

# Process malicious files
malicious_path = os.path.join(dataset_dir, "Malicious")
for folder in os.listdir(malicious_path):
    folder_path = os.path.join(malicious_path, folder)
    if os.path.isdir(folder_path):  # Ensure it's a folder
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and is_pdf(file_path):  # Check if it's really a PDF
                labels.append((file_path, 1))  # 1 = Malicious

# Save new dataset file
df = pd.DataFrame(labels, columns=["file_path", "label"])
df.to_csv("dataset_labels.csv", index=False)
print(f"✅ New dataset_labels.csv created! Total PDFs: {len(df)}")


#%% fast pre processing
import os
import pandas as pd
import fitz  # PyMuPDF
import math
import binascii
import multiprocessing
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
from collections import Counter
from tqdm import tqdm  # For progress bar

df = pd.read_csv("dataset_labels.csv")

# Function to compute entropy (optimized for speed)
def compute_entropy(file_path, sample_size=20000):
    """Compute entropy from first 20KB of the PDF file (for speed)."""
    with open(file_path, "rb") as f:
        data = f.read(sample_size)  # Read only first 20 KB
        if not data:
            return 0
        byte_counts = Counter(data)
        total_bytes = len(data)
        entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) for count in byte_counts.values())
        return entropy

# Feature extraction function (runs in parallel)
def extract_features(pdf_info):
    pdf_path, label, index, total = pdf_info

    try:
        # Print progress in console
        print(f"Processing file {index + 1}/{total}: {pdf_path}")

        doc = fitz.open(pdf_path)
        entropy = compute_entropy(pdf_path)

        # Count embedded fonts (fast method)
        font_count = sum(1 for page in doc if page.get_fonts() is not None for font in page.get_fonts()) if len(doc) > 0 else 0

        # Extract text and detect suspicious JavaScript
        try:
            pdf_text = extract_text(pdf_path)
        except (PDFSyntaxError, KeyError):
            return None  # Skip unreadable PDFs

        suspicious_js_keywords = ["eval", "unescape", "fromCharCode", "escape", "atob", "btoa", "decodeURIComponent"]
        js_count = sum(1 for word in suspicious_js_keywords if word in pdf_text)

        # Extract hexadecimal byte patterns
        with open(pdf_path, "rb") as f:
            hex_data = binascii.hexlify(f.read(10000)).decode("utf-8")  # Read only first 10 KB
        hex_count = sum(1 for pattern in ["25", "3C", "2F", "3E"] if pattern in hex_data)

        return (pdf_path, label, entropy, font_count, js_count, hex_count)

    except Exception as e:
        print(f"Skipping corrupted PDF: {pdf_path} | Error: {e}")
        return None  # Skip the problematic file

# Prepare list for multiprocessing
total_files = len(df)
pdf_info_list = [(row["file_path"], row["label"], i, total_files) for i, row in df.iterrows()]

# Run feature extraction in parallel with progress bar
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    results = list(tqdm(pool.imap(extract_features, pdf_info_list), total=total_files))

# Filter out None values (skipped PDFs)
valid_results = [r for r in results if r is not None]

# Create new DataFrame
df_filtered = pd.DataFrame(valid_results, columns=["file_path", "label", "entropy", "num_fonts", "num_suspicious_js", "num_hex_patterns"])

# Save optimized dataset
df_filtered.to_csv("pdf_features_optimized.csv", index=False)
print("🚀 Optimized feature extraction completed! Saved as pdf_features_optimized.csv")


#%% pre processing 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the optimized dataset
df = pd.read_csv("pdf_features_optimized.csv")

# Select features & labels
X = df.drop(columns=["file_path", "label"])  # Remove non-numerical columns
y = df["label"]  # 0 = Benign, 1 = Malicious

# Normalize features using Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Save preprocessed data for model training
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ Data preprocessing completed! Train and test datasets saved.")

#%% define the model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed datasets
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Define an improved MLP model
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),  # Input layer
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),  # Dropout to prevent overfitting
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation="sigmoid")  # Binary classification (0 or 1)
])

model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])


#%% Train the model
history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test))


#%% Evaluate on test data

from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

# Get predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Print results
print(f"\n📊 Test Accuracy: {test_acc:.4f}")
print(f"🔹 Precision: {precision_score(y_test, y_pred):.4f}")
print(f"🔹 Recall: {recall_score(y_test, y_pred):.4f}")
print(f"🔹 F1-Score: {f1_score(y_test, y_pred):.4f}")

# Note:
# Recall (Sensitivity)

#     Measures how many actual positive cases (malicious PDFs) were correctly detected.
#     Formula:
#     Recall=True Positives (TP)True Positives (TP)+False Negatives (FN)
#     Recall=True Positives (TP)+False Negatives (FN)True Positives (TP)​
#     High recall = Model detects almost all malware PDFs (few false negatives).
#     Low recall = Model misses many malware PDFs (many false negatives).

#  F1-Score

#     A balanced metric that considers both precision and recall.
#     Formula:
#     F1=2×Precision×RecallPrecision+Recall
#     F1=2×Precision+RecallPrecision×Recall​
#     High F1-score means the model has a good balance between detecting malware PDFs and avoiding false positives.
#     If F1-score is low, either recall or precision is weak.

# If recall is too low, the model is missing malware!
# If F1-score is high, the model is performing well overall.

#%% ploting result
import matplotlib.pyplot as plt

# Plot Training & Validation Accuracy
plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Accuracy Over Epochs")
plt.show()

# Plot Training & Validation Loss
plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Model Loss Over Epochs")
plt.show()


#%% Save the trained model

import joblib
joblib.dump(scaler, "scaler.pkl")  # Run this after fitting the scaler in training

model.save("model.keras")
print("✅ Model training completed and saved as model.keras")
