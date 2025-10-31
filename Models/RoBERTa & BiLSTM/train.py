import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import evaluate  # your existing evaluate.py

# =================== CONFIG ===================
DATA_PATH = r"D:\D Drive (Dhaval)\Research & Review Papers\NLP + DL Research Paper Misinformation Detection\Uni Modal\Final Project Implementation\Preprocessing\processed\merged_cleaned.csv"
MODEL_NAME = "distilroberta-base"   # smaller + faster
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 2
LR = 2e-5
USE_AMP = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =================== DATA ===================
df = pd.read_csv(DATA_PATH)
text_col, label_col = "cleaned_text", "label"
df = df.dropna(subset=[text_col, label_col])
df = df.sample(frac=0.2, random_state=42)  # use 20% data for fast run

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df[text_col].tolist(), df[label_col].tolist(), test_size=0.2, random_state=42
)

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = tokenizer(
            text, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)  # <-- fixed key
        }

train_ds = TextDataset(train_texts, train_labels)
val_ds = TextDataset(val_texts, val_labels)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

# =================== MODEL ===================
class RoBERTaBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(MODEL_NAME)
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256 * 4, 2)  # avg+max pooling concat

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # freeze RoBERTa for fast training
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        avg_pool = torch.mean(lstm_out, 1)
        max_pool, _ = torch.max(lstm_out, 1)
        out = torch.cat((avg_pool, max_pool), 1)
        out = self.dropout(out)
        return self.fc(out)

model = RoBERTaBiLSTM().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# =================== TRAIN ===================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_dl)
    print(f"Epoch {epoch+1} | Avg Train Loss: {avg_loss:.4f}")

# =================== SAVE MODEL ===================
MODEL_PATH = "roberta_bilstm_fast.pt"
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Training Complete. Model saved to {MODEL_PATH}")

# =================== EVALUATION ===================
print("\n=== Running Evaluation on Validation Set ===")
evaluate.evaluate_model(model, val_dl, DEVICE)
