import os
import torch
import librosa
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
from model import AudioCLIP  # Ensure correct path to AudioCLIP class

def inference(model, audio_path, device, sample_rate=22010, duration=4.0, class_names=None):
    """
    Perform inference on a single audio file or list of files.

    Args:
        model (nn.Module): The trained AudioCLIPWithHead model.
        audio_path (str or list): Path to a single audio file or list of audio file paths.
        device (torch.device): Device to run inference on.
        sample_rate (int): Target sample rate for audio files.
        duration (float): Duration in seconds to pad or truncate the audio.
        class_names (list): List of class names (optional).

    Returns:
        list: List of predicted class indices or class names if provided.
    """
    model.eval()  # Set the model to evaluation mode
    num_samples = int(sample_rate * duration)
    predictions = []

    # Ensure audio_path is a list
    if isinstance(audio_path, str):
        audio_path = [audio_path]

    for path in audio_path:
        # Load and preprocess the audio file
        waveform, _ = librosa.load(path, sr=sample_rate)

        # Pad or truncate waveform
        if len(waveform) < num_samples:
            waveform = np.pad(waveform, (0, num_samples - len(waveform)))
        else:
            waveform = waveform[:num_samples]

        # Normalize waveform between -1 and 1
        waveform = waveform / np.max(np.abs(waveform))

        # Convert to tensor and move to device
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)  # [1, num_samples]

        # Perform inference
        with torch.no_grad():
            outputs = model(audio=waveform_tensor)
            _, predicted_class = torch.max(outputs, 1)

        # Map to class name if provided
        predicted_class = predicted_class.item()
        if class_names:
            predictions.append(class_names[predicted_class])
        else:
            predictions.append(predicted_class)

        print(f"Audio: {path}, Predicted Class: {predictions[-1]}")

    return predictions


# Dataset Class
class UrbanSound8KDataset(Dataset):
    def __init__(self, csv_file, audio_dir, folds, sample_rate=16000, duration=4.0, transform=None):
        """
        UrbanSound8K Dataset class for loading audio data.
        Args:
            csv_file (str): Path to the metadata file.
            audio_dir (str): Path to the folder containing audio data.
            folds (list): List of folds to include in the dataset.
            sample_rate (int): Target sampling rate.
            duration (float): Duration to pad/truncate waveforms.
            transform (callable, optional): Optional transformations.
        """
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['fold'].isin(folds)]
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = os.path.join(self.audio_dir, f"fold{row['fold']}", row['slice_file_name'])
        waveform, _ = librosa.load(file_path, sr=self.sample_rate)

        # Pad or truncate waveform
        if len(waveform) < self.num_samples:
            waveform = np.pad(waveform, (0, self.num_samples - len(waveform)))
        else:
            waveform = waveform[:self.num_samples]

        # Normalize waveform between -1 and 1
        waveform = waveform / np.max(np.abs(waveform))
        if self.transform:
            waveform = self.transform(waveform)

        label = row['classID']
        return torch.tensor(waveform, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# AudioCLIP Model with Classification Head
class AudioCLIPWithHead(nn.Module):
    def __init__(self, pretrained, num_classes=10, device=None):
        super(AudioCLIPWithHead, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audioclip = AudioCLIP(pretrained=pretrained, multilabel=False)

        # Freeze all parameters except audio-related ones
        for p in self.audioclip.parameters():
            p.requires_grad = False
        for p in self.audioclip.audio.parameters():
            p.requires_grad = True

        #self.classification_head = nn.Linear(1024, num_classes)
        self.classification_head = nn.Sequential(
            nn.Linear(1024, 256),  # First hidden layer
            nn.ReLU(),             # Non-linearity
            nn.Dropout(0.5),       # Dropout for regularization
            nn.Linear(256, num_classes)  # Output layer
        )


    def forward(self, audio):
        # Extract audio features
        audio_features = self.audioclip.encode_audio(audio=audio)
        # Get audio features
        # ((audio_features, _, _), _), _ = self.audioclip(
        #     audio=audio,
        #     batch_indices=torch.arange(audio.shape[0], dtype=torch.int64, device=device)
        # )
        # audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)  # Normalize
        # Pass through classification head
        output = self.classification_head(audio_features)
        return output


# DataLoader Utility
def get_loaders(csv_file, audio_dir, train_folds, val_folds, test_folds, batch_size):
    train_dataset = UrbanSound8KDataset(csv_file, audio_dir, train_folds)
    val_dataset = UrbanSound8KDataset(csv_file, audio_dir, val_folds)
    test_dataset = UrbanSound8KDataset(csv_file, audio_dir, test_folds)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader


# Training Function
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10, save_path="audioclip_model.pth"):
    model.to(device)
    best_val_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(audio=data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")

        val_accuracy = validate_model(model, val_loader, device, criterion)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"Saving model with validation accuracy: {best_val_accuracy:.4f}")
            torch.save(model.state_dict(), save_path)


# Validation Function
def validate_model(model, val_loader, device, criterion):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc="Validating"):
            data, labels = data.to(device), labels.to(device)
            outputs = model(audio=data)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.4f}")
    return val_accuracy


# Evaluation Function
def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Evaluating"):
            data, labels = data.to(device), labels.to(device)
            outputs = model(audio=data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))


# Main Script
if __name__ == "__main__":
    # Paths and Configurations
    csv_file = "/data/urbansound8k/UrbanSound8K.csv"
    audio_dir = "/data/urbansound8k/"
    train_folds, val_folds, test_folds = [1, 2, 3, 4, 5, 6, 7, 8], [10], [10]
    batch_size, epochs, learning_rate = 32, 20, 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader, test_loader = get_loaders(csv_file, audio_dir, train_folds, val_folds, test_folds, batch_size)

    # Initialize Model
    model = AudioCLIPWithHead(pretrained="/home/ilias/projects/AudioCLIP/assets/UrbanSound8K_Multimodal-Audio-x2_ACLIP-CV01/UrbanSound8K_Multimodal-Audio-x2_ACLIP-CV01_ACLIP-CV01_performance=0.9188.pt", num_classes=10, device=device)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train and Evaluate
    train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_path="best_model.pth")
    
    # Load the Best Model
    print("Loading the best saved model...")
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)

    # Evaluate the Model on the Test Set
    #print("Evaluating the model on the test set...")
    evaluate_model(model, test_loader, device)

    class_names = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
    audio_file = "/home/ilias/projects/adversarial_thesis/data/adv_2_dog_bark.wav"
    predictions = inference(model, audio_file, device, class_names=class_names)

    # Print results
    print("Predicted Class:", predictions[0])