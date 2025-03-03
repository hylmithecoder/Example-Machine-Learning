from transformers import GPT2Tokenizer

def load_tokenizer(model_name: str = "gpt2"):
    """Muat tokenizer GPT-2 dan set padding token."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Atur token padding
    return tokenizer

def tokenize_texts(texts: list, tokenizer, max_length: int = 128):
    """Tokenisasi batch teks."""
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"  # Return PyTorch tensor
    )
