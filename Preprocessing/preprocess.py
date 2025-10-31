import pandas as pd
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Paths to datasets (your format)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAKE_PATH = os.path.join(BASE_DIR, '../ISOT Dataset/Fake.csv')
TRUE_PATH = os.path.join(BASE_DIR, '../ISOT Dataset/True.csv')

# Output directory for cleaned data
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Load and process dataset
def load_and_process(path, label):
    df = pd.read_csv(path)
    df = df[['title', 'text', 'subject', 'date']]
    df['combined_text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    df['label'] = label
    return df[['cleaned_text', 'label']]

# Process both datasets
fake_df = load_and_process(FAKE_PATH, 0)
true_df = load_and_process(TRUE_PATH, 1)

# Save individual cleaned files
fake_df.to_csv(os.path.join(OUTPUT_DIR, 'fake_cleaned.csv'), index=False)
true_df.to_csv(os.path.join(OUTPUT_DIR, 'true_cleaned.csv'), index=False)

# Merge, shuffle, and save merged file
merged_df = pd.concat([fake_df, true_df], ignore_index=True).sample(frac=1, random_state=42)
merged_df.to_csv(os.path.join(OUTPUT_DIR, 'merged_cleaned.csv'), index=False)

print("[✅] Preprocessing complete.")
print(f"[📁] Cleaned data saved in: {OUTPUT_DIR}")
