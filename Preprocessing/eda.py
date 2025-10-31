import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
import os

# Download required NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Paths to datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAKE_PATH = os.path.join(BASE_DIR, '../ISOT Dataset/Fake.csv')
TRUE_PATH = os.path.join(BASE_DIR, '../ISOT Dataset/True.csv')

# Create output directory if it doesn't exist
output_dir = os.path.join(BASE_DIR, "EDA Analysis")
os.makedirs(output_dir, exist_ok=True)

# Load datasets
fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

# Add labels
true_df['label'] = 1
fake_df['label'] = 0

# Merge datasets
df = pd.concat([true_df, fake_df], ignore_index=True)
df['text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)
df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

# Plot: Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label', palette='Set2')
plt.title('Class Distribution (Fake = 0, True = 1)')
plt.xticks([0, 1], ['Fake', 'True'])
plt.xlabel('News Type')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Class_Distribution.png"))
plt.close()

# Plot: Word count distribution
plt.figure(figsize=(10, 6))
sns.histplot(df[df['label'] == 0]['word_count'], bins=50, label='Fake', color='red', kde=True)
sns.histplot(df[df['label'] == 1]['word_count'], bins=50, label='True', color='blue', kde=True)
plt.legend()
plt.title('Word Count Distribution')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Word_Count_Distribution.png"))
plt.close()

# Generate word cloud
def generate_wordcloud(text_data, title, color, filename):
    text = ' '.join(text_data)
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=stop_words, colormap=color).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

generate_wordcloud(df[df['label'] == 0]['clean_text'], 'Fake News Word Cloud', 'Reds', 'Fake_WordCloud.png')
generate_wordcloud(df[df['label'] == 1]['clean_text'], 'True News Word Cloud', 'Blues', 'True_WordCloud.png')

# Most frequent words
def plot_top_words(data, label, n=20):
    words = ' '.join(data[data['label'] == label]['clean_text']).split()
    filtered_words = [word for word in words if word not in stop_words]
    word_freq = Counter(filtered_words).most_common(n)
    words, counts = zip(*word_freq)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts), y=list(words), palette='viridis')
    plt.title(f'Top {n} Words - {"Fake" if label == 0 else "True"} News')
    plt.xlabel('Frequency')
    plt.tight_layout()
    filename = f'{"Fake" if label == 0 else "True"}_Top_Words.png'
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_top_words(df, label=0)
plot_top_words(df, label=1)

# Bigrams
def plot_top_bigrams(data, label, n=20):
    text = ' '.join(data[data['label'] == label]['clean_text'])
    tokens = [word for word in text.split() if word not in stop_words]
    bigrams = ngrams(tokens, 2)
    bigram_freq = Counter(bigrams).most_common(n)
    bigram_words = [' '.join(bigram) for bigram, freq in bigram_freq]
    counts = [freq for bigram, freq in bigram_freq]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=bigram_words, palette='magma')
    plt.title(f'Top {n} Bigrams - {"Fake" if label == 0 else "True"} News')
    plt.xlabel('Frequency')
    plt.tight_layout()
    filename = f'{"Fake" if label == 0 else "True"}_Top_Bigrams.png'
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_top_bigrams(df, label=0)
plot_top_bigrams(df, label=1)
