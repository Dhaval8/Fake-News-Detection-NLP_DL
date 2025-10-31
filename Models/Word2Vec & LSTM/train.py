# train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec

# ============ Training with Custom Word2Vec ============
def train_with_custom_word2vec():
    print("🚀 Training with custom Word2Vec model...")

    # Import your modules
    try:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.utils import pad_sequences
        from model import LSTM_Word2Vec
        from utils import NewsDataset
        from evaluate import evaluate_model
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "../../Preprocessing/processed/merged_cleaned.csv")

    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Dataset loaded: {df.shape}")

        # ✅ Clean labels: drop NaN and force int
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            df["cleaned_text"], df["label"], test_size=0.2, random_state=42
        )

    except Exception as e:
        print(f"❌ Error loading/cleaning data: {e}")
        return

    # Tokenize for Word2Vec training
    print("🔤 Preparing text for Word2Vec training...")
    train_sentences = [text.split() for text in X_train if isinstance(text, str)]

    # Train custom Word2Vec model
    print("🧠 Training custom Word2Vec model...")
    word2vec_model = Word2Vec(
        sentences=train_sentences,
        vector_size=300,  # embedding dimension
        window=5,
        min_count=2,
        workers=4,
        epochs=10,
        sg=1,  # skip-gram
    )

    print(f"✅ Word2Vec trained! Vocabulary size: {len(word2vec_model.wv)}")

    # Save custom Word2Vec
    custom_w2v_path = os.path.join(BASE_DIR, "custom_word2vec.model")
    word2vec_model.save(custom_w2v_path)
    print(f"💾 Custom Word2Vec saved to: {custom_w2v_path}")

    # Tokenize texts for LSTM
    tokenizer = Tokenizer(num_words=50000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_train), maxlen=300, padding="post"
    )
    X_val_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_val), maxlen=300, padding="post"
    )

    # Build embedding matrix from Word2Vec
    def build_custom_embedding_matrix(tokenizer, word2vec_model):
        vocab_size = len(tokenizer.word_index) + 1
        embedding_dim = word2vec_model.vector_size
        embedding_matrix = np.zeros((vocab_size, embedding_dim))

        hits = 0
        misses = 0

        for word, i in tokenizer.word_index.items():
            if word in word2vec_model.wv:
                embedding_matrix[i] = word2vec_model.wv[word]
                hits += 1
            else:
                embedding_matrix[i] = np.random.normal(
                    scale=0.1, size=(embedding_dim,)
                )
                misses += 1

        print(
            f"Embedding coverage: {hits}/{hits+misses} ({hits/(hits+misses)*100:.1f}%)"
        )
        return embedding_matrix

    embedding_matrix = build_custom_embedding_matrix(tokenizer, word2vec_model)

    # Create datasets
    train_dataset = NewsDataset(X_train_seq, y_train.tolist())
    val_dataset = NewsDataset(X_val_seq, y_val.tolist())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize model
    model = LSTM_Word2Vec(embedding_matrix).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    print("\n🚀 Starting LSTM training...")
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")
        model.train()
        total_loss = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Average Loss: {avg_loss:.4f}")

        # Validation
        val_accuracy = evaluate_model(model, val_loader, device)
        print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(BASE_DIR, "lstm_custom_word2vec_model.pth"))
    print("✅ Model saved!")


# ============ Training with Random Embeddings (Quick Test) ============
def train_with_random_embeddings():
    print("🎲 Training with random embeddings...")

    try:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.utils import pad_sequences
        from model import LSTM_Word2Vec
        from utils import NewsDataset
        from evaluate import evaluate_model
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "../../Preprocessing/processed/merged_cleaned.csv")

    df = pd.read_csv(DATA_PATH)

    # ✅ Clean labels
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        df["cleaned_text"], df["label"], test_size=0.2, random_state=42
    )

    # Tokenization
    tokenizer = Tokenizer(num_words=50000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_train), maxlen=300, padding="post"
    )
    X_val_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_val), maxlen=300, padding="post"
    )

    # Random embedding matrix
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 300
    embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim))
    print(f"Created random embedding matrix: {embedding_matrix.shape}")

    # Dataset + loaders
    train_dataset = NewsDataset(X_train_seq, y_train.tolist())
    val_dataset = NewsDataset(X_val_seq, y_val.tolist())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model
    model = LSTM_Word2Vec(embedding_matrix).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")
        model.train()
        total_loss = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_accuracy = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {val_accuracy:.4f}")

    torch.save(model.state_dict(), os.path.join(BASE_DIR, "lstm_random_embeddings_model.pth"))
    print("✅ Random embedding model saved!")


if __name__ == "__main__":
    print("Choose your option:")
    print("1. Train custom Word2Vec on your data (recommended)")
    print("2. Use random embeddings (quick test)")

    choice = input("Enter choice (1 or 2): ")

    if choice == "1":
        train_with_custom_word2vec()
    elif choice == "2":
        train_with_random_embeddings()
    else:
        print("Invalid choice!")
