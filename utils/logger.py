import csv
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs", filename="training_log.csv"):
        self.log_dir = log_dir
        self.filename = filename
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)
        
        # Jika file tidak ada, buat header
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Epoch", "Loss", "Accuracy"])
    
    def log(self, epoch, loss, accuracy=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, epoch, loss, accuracy if accuracy is not None else "N/A"])
        
        print(f"[LOG] Epoch {epoch} - Loss: {loss:.4f} - Accuracy: {accuracy if accuracy is not None else 'N/A'}")
