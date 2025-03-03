import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.dataset import load_dataset_subset, preprocess_data
from models.transformer import DecoderOnlyTransformer
from utils.tokenizer import load_tokenizer
from utils.logger import Logger  # Import class Logger

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Buat instance logger
logger = Logger()

def train():
    print("Loading dataset...")
    texts = load_dataset_subset()
    input_ids = preprocess_data(texts)
    dataset = TensorDataset(input_ids)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Initializing model...")
    tokenizer = load_tokenizer()
    model = DecoderOnlyTransformer(vocab_size=tokenizer.vocab_size, d_model=512, num_layers=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch[0].to(device)
            
            outputs = model(input_ids[:, :-1])
            loss = criterion(outputs.contiguous().view(-1, tokenizer.vocab_size), input_ids[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        logger.log(epoch, avg_loss)  # Simpan log ke CSV
        
        print(f"Epoch {epoch}/{EPOCHS}, Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "trained_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
