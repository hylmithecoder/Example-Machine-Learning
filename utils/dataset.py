from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from utils.tokenizer import load_tokenizer

def load_dataset_subset(dataset_name="roneneldan/TinyStories", subset_size=100000):
    dataset = load_dataset(dataset_name, split="train")  # Hapus slicing di sini
    dataset = dataset.select(range(subset_size))  # Gunakan .select() untuk memilih subset
    return [item["text"] for item in dataset]

def preprocess_data(texts, max_length=512):
    tokenizer = load_tokenizer()
    tokenized_data = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")
    return tokenized_data["input_ids"]

class TextDataset(Dataset):
    def __init__(self, dataset_name="roneneldan/TinyStories", subset_size=100000, max_length=512):
        self.tokenizer = load_tokenizer()
        self.dataset = load_dataset_subset(dataset_name, subset_size)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]
        tokenized = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0)
        }

def get_dataloader(batch_size=16, shuffle=True):
    dataset = TextDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
