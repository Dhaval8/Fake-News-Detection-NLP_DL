import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class CustomTextDataset(Dataset):
    """
    Returns dictionaries with keys:
      - input_ids
      - attention_mask
      - labels
    (labels is a torch.long)
    """

    def __init__(self, texts, labels, model_name='distilroberta-base', max_len=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
