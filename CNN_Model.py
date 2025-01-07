# CNN_Model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
from sklearn.model_selection import KFold
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import gc


def print_gpu_memory():
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


class GameFrameDataset(Dataset):
    def __init__(self, video_dir: str, frames_per_segment: int = 16, segments_per_video: int = 10):
        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
        self.frames_per_segment = frames_per_segment
        self.segments_per_video = segments_per_video
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"\nDataset initialized with {len(self.video_paths)} videos")
        print(f"Frames per segment: {frames_per_segment}")
        print(f"Segments per video: {segments_per_video}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        try:
            video_path = self.video_paths[idx]
            frames = []

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video: {video_path}")
                return self._get_default_tensor()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return self._get_default_tensor()

            # Calculate frame indices for uniform sampling
            segment_size = max(1, total_frames // self.segments_per_video)
            frame_indices = []

            for segment_idx in range(self.segments_per_video):
                segment_start = segment_idx * segment_size
                segment_end = min(segment_start + segment_size, total_frames)
                segment_indices = np.linspace(segment_start, segment_end - 1, self.frames_per_segment, dtype=int)
                frame_indices.extend(segment_indices)

            # Read frames
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
                frames.append(frame)

            cap.release()

            # Handle case where we didn't get enough frames
            while len(frames) < (self.frames_per_segment * self.segments_per_video):
                frames.append(self._get_empty_frame())

            frames_tensor = torch.stack(frames)
            frames_tensor = frames_tensor.view(self.segments_per_video, self.frames_per_segment, 3, 224, 224)

            combat_score = 1.0 if "combat" in video_path.lower() or "boss" in video_path.lower() else 0.0
            label = torch.tensor([combat_score], dtype=torch.float32)

            return frames_tensor, label

        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return self._get_default_tensor()

    def _get_empty_frame(self):
        return torch.zeros(3, 224, 224, dtype=torch.float32)

    def _get_default_tensor(self):
        frames = torch.zeros(self.segments_per_video, self.frames_per_segment, 3, 224, 224, dtype=torch.float32)
        label = torch.tensor([0.0], dtype=torch.float32)
        return frames, label


class MemoryEfficientGameCNN(nn.Module):
    def __init__(self, frames_per_segment=16, segments_per_video=10, device='cuda'):
        super(MemoryEfficientGameCNN, self).__init__()

        # Frame feature extractor (ResNet-18 with gradient checkpointing)
        resnet = models.resnet18(pretrained=True)
        self.frame_encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Enable gradient checkpointing for memory efficiency
        for module in self.frame_encoder.modules():
            if isinstance(module, nn.Module):
                module.register_forward_hook(self._hook_fn)

        # Temporal encoder with reduced parameters
        self.segment_encoder = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # Lightweight sequence encoder
        self.sequence_encoder = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Efficient classifier (removed Sigmoid for BCEWithLogitsLoss)
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)  # Removed Sigmoid layer
        )

        self.device = device
        self.to(device)

    @staticmethod
    def _hook_fn(module, input, output):
        if torch.is_tensor(output):
            output.detach_()
        elif isinstance(output, tuple):
            output = tuple(o.detach_() for o in output if torch.is_tensor(o))

    def forward(self, x, chunk_size=4):
        batch_size, num_segments, frames_per_segment, C, H, W = x.shape

        # Process frames in chunks to save memory
        frame_features = []
        x_reshaped = x.view(-1, C, H, W)

        for i in range(0, len(x_reshaped), chunk_size):
            chunk = x_reshaped[i:i + chunk_size]
            with torch.cuda.amp.autocast():
                features = self.frame_encoder(chunk)
            frame_features.append(features.cpu())  # Move to CPU to save GPU memory
            del features
            torch.cuda.empty_cache()

        x = torch.cat(frame_features, dim=0).to(self.device)
        del frame_features
        torch.cuda.empty_cache()

        # Reshape and process through temporal encoder
        _, C, H, W = x.shape
        x = x.view(batch_size * num_segments, frames_per_segment, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        x = self.segment_encoder(x)
        x = x.view(batch_size, num_segments, -1)

        # Process sequence
        x, _ = self.sequence_encoder(x)
        x = torch.mean(x, dim=1)

        # Classification
        x = self.classifier(x)

        return x


def train_model(model, train_loader, val_loader, num_epochs=30, device='cuda', checkpoint_dir='checkpoints'):
    print("\nStarting training...")
    os.makedirs(checkpoint_dir, exist_ok=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    # Load checkpoint if exists
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming from epoch {start_epoch}")

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        last_epoch=-1 if start_epoch == 0 else start_epoch * len(train_loader)
    )

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0

        # Training phase
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            try:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Clear memory
                torch.cuda.empty_cache()
                gc.collect()

                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()

                current_loss = loss.item()
                train_loss += current_loss
                batch_count += 1

                # Clear memory
                del outputs, loss
                torch.cuda.empty_cache()

                train_pbar.set_postfix({'loss': f'{current_loss:.4f}'})

                if batch_idx % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': train_loss / batch_count
                    }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pth'))

            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                try:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        batch_loss = criterion(outputs, targets)

                    val_loss += batch_loss.item()
                    val_batch_count += 1

                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                    del outputs, batch_loss
                    torch.cuda.empty_cache()

                except RuntimeError as e:
                    print(f"Error in validation: {str(e)}")
                    continue

        # Calculate metrics
        avg_train_loss = train_loss / batch_count if batch_count > 0 else float('inf')
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        accuracy = 100 * correct / total if total > 0 else 0

        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')
        print_gpu_memory()

        # Save checkpoints
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss,
            'best_val_loss': best_val_loss,
            'accuracy': accuracy
        }

        torch.save(checkpoint_data, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, 'best_model.pth'))
            print('Saved best model checkpoint')

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            periodic_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(checkpoint_data, periodic_path)
            print(f'Saved periodic checkpoint to {periodic_path}')


def main():
    FRAMES_PER_SEGMENT = 16
    SEGMENTS_PER_VIDEO = 10
    BATCH_SIZE = 2
    CHECKPOINT_DIR = 'training_checkpoints'
    NUM_EPOCHS = 30

    # Set device and clear cache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    torch.cuda.empty_cache()
    gc.collect()

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Initialize dataset
    dataset = GameFrameDataset(
        "witcher3_dataset",
        frames_per_segment=FRAMES_PER_SEGMENT,
        segments_per_video=SEGMENTS_PER_VIDEO
    )

    # Print dataset statistics
    combat_count = sum(1 for path in dataset.video_paths if "combat" in path.lower() or "boss" in path.lower())
    print(f"\nDataset Statistics:")
    print(f"Total videos: {len(dataset)}")
    print(f"Combat/Boss videos: {combat_count}")
    print(f"Non-combat videos: {len(dataset) - combat_count}")

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"\nSplit sizes:")
    print(f"Training set: {len(train_dataset)} videos")
    print(f"Validation set: {len(val_dataset)} videos")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print("\nInitializing model...")
    model = MemoryEfficientGameCNN(
        frames_per_segment=FRAMES_PER_SEGMENT,
        segments_per_video=SEGMENTS_PER_VIDEO,
        device=device
    )

    # Print model summary
    print("\nModel Architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nStarting training process...")
    train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR
    )


if __name__ == "__main__":
    main()