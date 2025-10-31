import numpy as np
import torch
from torch.utils.data import Dataset

# Fixed imports for TensorFlow 2.x
try:
    from gensim.models import KeyedVectors
except ImportError:
    print("Please install gensim: pip install gensim")
    raise

def load_word2vec_model(path):
    """Load Word2Vec model from binary file"""
    try:
        print(f"Loading Word2Vec from: {path}")
        model = KeyedVectors.load_word2vec_format(path, binary=True)
        print(f"Word2Vec loaded successfully. Vector size: {model.vector_size}")
        return model
    except Exception as e:
        print(f"Error loading Word2Vec model: {e}")
        raise

def build_embedding_matrix(tokenizer, word2vec):
    """Build embedding matrix from tokenizer and word2vec model"""
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = word2vec.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    hits = 0
    misses = 0
    
    for word, i in tokenizer.word_index.items():
        if word in word2vec:
            embedding_matrix[i] = word2vec[word]
            hits += 1
        else:
            misses += 1
    
    print(f"Embedding matrix built: {hits} hits, {misses} misses")
    print(f"Coverage: {hits / (hits + misses) * 100:.2f}%")
    
    return embedding_matrix

class NewsDataset(Dataset):
    """Custom Dataset class for news text classification"""
    
    def __init__(self, sequences, labels):
        """
        Args:
            sequences: List or array of tokenized sequences
            labels: List or array of labels
        """
        self.sequences = sequences
        self.labels = labels
        
        # Convert to numpy arrays if they aren't already
        if not isinstance(sequences, np.ndarray):
            self.sequences = np.array(sequences)
        if not isinstance(labels, (list, np.ndarray)):
            self.labels = list(labels)
            
        print(f"Dataset created with {len(self.sequences)} samples")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label