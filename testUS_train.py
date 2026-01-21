import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

import torch.nn as nn
import torch.optim as optim

# --- 1. Dataset Class ---
class UltrasoundDataset(Dataset):
    def __init__(self, csv_file):
        self.data_info = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        npy_path = self.data_info.iloc[idx]['npy_path']
        label = self.data_info.iloc[idx]['label']

        # Load numpy file
        # Expected shape: (2, 32, 9000)
        try:
            sample = np.load(npy_path)
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            # Return zeros in case of error to keep training going, or handle specifically
            sample = np.zeros((32, 9000), dtype=np.float32)

        # Convert to tensor
        sample = sample[0, :, :] #using first channel only  
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label

# --- 2. Model Architecture ---
class UltrasoundClassifier(nn.Module):
    def __init__(self, num_classes):
        super(UltrasoundClassifier, self).__init__()
        
        # Input shape: (Batch, 32, 9000)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Block 3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling -> (Batch, 128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 3. Training Loop ---
def train_model():
    # Configuration
    csv_path = r'E:\dataset\ultrasound_video_audio\DATA\dataset\train_dataset.csv'
    batch_size = 1024
    num_epochs = 200
    learning_rate = 1e-3
    print("cuda available:", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset
    full_dataset = UltrasoundDataset(csv_path)
    
    # Determine number of classes automatically
    unique_labels = full_dataset.data_info['label'].unique()
    num_classes = len(unique_labels)
    print(f"Detected {num_classes} classes: {sorted(unique_labels)}")

    # Split dataset (Optional: split into train/val)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)

    # Initialize Model, Loss, Optimizer
    model = UltrasoundClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Training Loop
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': running_loss / (total/batch_size), 'acc': 100 * correct / total})

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_loss = val_running_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        # Ensure output directory exists
        os.makedirs('./outputs', exist_ok=True)

        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            scheduler.step()
            checkpoint_path = f'./outputs/model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './outputs/best_model.pth')
            print(f"New best model saved with Val Loss: {val_loss:.4f}")

    # Save Model
    torch.save(model.state_dict(), 'ultrasound_model.pth')
    print("Training Complete. Model saved.")   

def test_model():
    # Configuration
    # test_csv_path = r'E:\dataset\ultrasound_video_audio\DATA\dataset\test_dataset.csv'
    test_csv_path = r'E:\dataset\ultrasound_video_audio\DATA\dataset\test_dataset_ldxTest.csv'

    model_path = './outputs/best_model.pth' # Using best model found during training
    # model_path = './outputs/model_epoch_10.pth' # Using best model found during training

    output_txt_path = './outputs/test_results.txt'
    batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(test_csv_path):
        print(f"Error: Test dataset not found at {test_csv_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}. Please train first.")
        return

    print(f"Test: Using device: {device}")

    # Load Dataset
    # IMPORTANT: shuffle=False to ensure predictions align with the dataframe order for reporting
    test_dataset = UltrasoundDataset(test_csv_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load Model Checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Automatically determine num_classes from the saved model weights
    # The classifier's last layer is index 4 in the Sequential block: Linear(64, num_classes)
    if 'classifier.4.weight' in checkpoint:
        num_classes = checkpoint['classifier.4.weight'].shape[0]
        print(f"Inferred {num_classes} classes from model checkpoint.")
    else:
        # Fallback if key structure varies
        unique_labels = test_dataset.data_info['label'].unique()
        num_classes = len(unique_labels)
        print(f"Using {num_classes} classes from dataset (Fallback).")

    # Initialize Model
    model = UltrasoundClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    all_preds = []
    all_labels = []

    print("Starting Inference...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate Accuracy
    total = len(all_labels)
    correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
    test_acc = 100 * correct / total if total > 0 else 0
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Save results to TXT
    # We use test_dataset.data_info directly as order is preserved
    df = test_dataset.data_info
    
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Test finished at {pd.Timestamp.now()}\n")
            f.write(f"Total Accuracy: {test_acc:.2f}%\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Npy_Path':<60} {'True':<10} {'Pred':<10} {'Result'}\n")
            f.write("-" * 80 + "\n")
            
            for i in range(len(df)):
                # Ensure we don't go out of bounds if dataset size mismatches loader iteration (rare)
                if i >= len(all_preds): break
                
                path = df.iloc[i]['npy_path']
                true_lbl = df.iloc[i]['label']
                pred_lbl = all_preds[i]
                result = "Correct" if true_lbl == pred_lbl else "Wrong"
                
                f.write(f"{path:<60} {true_lbl:<10} {pred_lbl:<10} {result}\n")
        
        print(f"Detailed classification results saved to {output_txt_path}")
        
    except Exception as e:
        print(f"Error saving results: {e}")



if __name__ == "__main__":
    # train_model()
    test_model()