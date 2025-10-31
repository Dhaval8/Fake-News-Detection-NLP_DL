import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import os
import sys
from tqdm import tqdm

# Import custom modules
from model import BERT_CNN
from utils import TextDataset, collate_fn
from evaluate import evaluate_model

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
DATA_PATH = r"D:\D Drive (Dhaval)\Research & Review Papers\NLP + DL Research Paper Misinformation Detection\Uni Modal\Final Project Implementation\Preprocessing\processed\merged_cleaned.csv"

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded. Shape: {df.shape}")

# ✅ Clean labels
df = df.dropna(subset=["label"])          # drop rows with NaN labels
df["label"] = df["label"].astype(int)     # ensure integer labels

texts = df["cleaned_text"].astype(str).tolist()
labels = df["label"].tolist()

print(f"Unique labels: {set(labels)}")

# Tokenizer and dataset preparation
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length=128)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)

# Model, loss, optimizer
model = BERT_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training loop
for epoch in range(3):
    print(f"\n{'='*50}")
    print(f"EPOCH {epoch+1}/3")
    print(f"{'='*50}")
    
    model.train()
    total_loss = 0
    train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

    for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_pbar):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_train_loss = total_loss / len(train_loader)
    print(f"Average training loss: {avg_train_loss:.4f}")

    val_accuracy = evaluate_model(model, val_loader, device)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

# Save model
torch.save(model.state_dict(), "bert_cnn.pth")
print("✅ Model saved as bert_cnn.pth")
