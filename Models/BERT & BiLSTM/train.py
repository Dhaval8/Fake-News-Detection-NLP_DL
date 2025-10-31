import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import os

from model import BERT_BiLSTM
from evaluate import evaluate_model
from utils import NewsDataset

# === PATH SETUP (Your Format) === #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_DATA_PATH = os.path.join(BASE_DIR, '../../Preprocessing/processed/merged_cleaned.csv')

# === TRAIN FUNCTION === #
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load preprocessed data
    df = pd.read_csv(MERGED_DATA_PATH)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)

    # Load tokenizer and datasets
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = NewsDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    val_dataset = NewsDataset(X_val.tolist(), y_val.tolist(), tokenizer)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Model, Loss, Optimizer
    model = BERT_BiLSTM().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Training Loss: {total_loss / len(train_loader):.4f}")

    # Save model
    model_save_path = os.path.join(BASE_DIR, "bert_bilstm_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"[✅] Model saved at {model_save_path}")

    # Evaluate on validation set
    evaluate_model(model, val_loader, device)

if __name__ == "__main__":
    train()
