import os
import torch
import librosa
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np
import multiprocessing as mp

# Function to process audio
def process_audio(file_path, sample_rate, num_samples):
    try:
        waveform, sr = librosa.load(file_path, sr=sample_rate)
        if len(waveform) < num_samples:
            waveform = np.pad(waveform, (0, num_samples - len(waveform)))
        else:
            waveform = waveform[:num_samples]
        return os.path.basename(file_path), waveform
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return os.path.basename(file_path), None


class UrbanSound8KDataset(Dataset):
    def __init__(self, root, folds, sample_rate=22050, duration=4.0, transform=None):
        self.root = root
        self.folds = folds
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.transform = transform
        self.data = []
        self.class_idx_to_label = {}  # Initialize mappings
        self.label_to_class_idx = {}
        self._prepare_data()

    def _prepare_data(self):
        meta = pd.read_csv(os.path.join(self.root, "UrbanSound8K.csv"))
        meta = meta[meta["fold"].isin(self.folds)]

        # Generate class mappings
        self.class_idx_to_label = meta.set_index("classID")["class"].to_dict()
        self.label_to_class_idx = {label: idx for idx, label in self.class_idx_to_label.items()}

        file_paths = [
            (os.path.join(self.root, f"fold{row['fold']}", row["slice_file_name"]), self.sample_rate, self.num_samples)
            for _, row in meta.iterrows()
        ]

        with mp.Pool(processes=os.cpu_count()) as pool:
            for fn, wav in tqdm(
                pool.starmap(process_audio, file_paths),
                desc=f"Loading UrbanSound8K (train={len(self.folds) > 1})",
                total=len(file_paths)
            ):
                if wav is not None:
                    row = meta.loc[meta["slice_file_name"] == fn].iloc[0]
                    self.data.append({
                        "audio": wav,
                        "label": row["classID"],
                        "category": row["class"],
                        "fold": row["fold"]
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        audio = sample["audio"]
        label = sample["label"]

        if self.transform:
            audio = self.transform(audio)

        return torch.tensor(audio, dtype=torch.float32), torch.tensor(label, dtype=torch.long)



# Data loader utility
def get_loaders(root, train_folds, val_folds, test_folds, batch_size=32, sample_rate=22050, duration=4.0, transform=None):
    train_dataset = UrbanSound8KDataset(root, folds=train_folds, sample_rate=sample_rate, duration=duration, transform=transform)
    val_dataset = UrbanSound8KDataset(root, folds=val_folds, sample_rate=sample_rate, duration=duration, transform=transform)
    test_dataset = UrbanSound8KDataset(root, folds=test_folds, sample_rate=sample_rate, duration=duration, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader


def evaluate_model(model, eval_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    # Precompute text features for UrbanSound8K class labels
    class_idx_to_label = eval_loader.dataset.class_idx_to_label
    label_to_class_idx = eval_loader.dataset.label_to_class_idx

    # Generate text labels as expected by the pretrained model
    text_labels = [
        [class_idx_to_label[class_idx]]
        for class_idx in sorted(class_idx_to_label.keys())
    ]

    # Encode text labels
    text_features = model.encode_text(text_labels).to(device)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize
    text_features = text_features.unsqueeze(1).transpose(0, 1)  # Prepare for matrix multiplication

    logit_scale_at = torch.clamp(model.logit_scale_at.exp(), min=1.0, max=100.0)

    with torch.no_grad():
        for audio, labels in tqdm(eval_loader, desc="Evaluating"):
            audio = audio.to(device)
            labels = labels.to(device)

            # Get audio features
            ((audio_features, _, _), _), _ = model(
                audio=audio,
                batch_indices=torch.arange(audio.shape[0], dtype=torch.int64, device=device)
            )
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)  # Normalize
            audio_features = audio_features.unsqueeze(1)

            # Compute logits (audio-to-text alignment)
            y_pred_logits = (logit_scale_at * audio_features @ text_features.transpose(-1, -2)).squeeze(1)

            # Single-label classification
            y_pred_batch = torch.softmax(y_pred_logits, dim=-1).argmax(dim=-1).cpu().numpy()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(y_pred_batch)

    # Single-label evaluation
    correct = sum(int(y_t == y_p) for y_t, y_p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    target_names = [class_idx_to_label[class_idx] for class_idx in sorted(class_idx_to_label.keys())]
    report = classification_report(y_true, y_pred, target_names=target_names)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(report)
    return accuracy, report



if __name__ == "__main__":
    # Configurations
    root = "/data/urbansound8k/"
    train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
    val_folds = [9]
    test_folds = [10]
    batch_size = 32

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader = get_loaders(root, train_folds, val_folds, test_folds, batch_size)
    print(f"Class mappings: {test_loader.dataset.class_idx_to_label}")
    print(f"Expected number of classes: {len(test_loader.dataset.class_idx_to_label)}")


    # Load the pretrained AudioCLIP model
    from model import AudioCLIP  # Ensure this is your AudioCLIP implementation
    model = AudioCLIP(pretrained="/home/ilias/projects/AudioCLIP/assets/UrbanSound8K_Multimodal-Audio-x2_ACLIP-CV01/UrbanSound8K_Multimodal-Audio-x2_ACLIP-CV01_ACLIP-CV01_performance=0.9247.pt", multilabel=False)
    model.to(device)

    # Freeze all parameters (only needed if you plan to fine-tune)
    for p in model.parameters():
        p.requires_grad = False

    # Evaluate the pretrained model
    evaluate_model(model, test_loader, device)
