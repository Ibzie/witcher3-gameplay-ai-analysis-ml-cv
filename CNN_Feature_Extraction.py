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
        video_path = self.video_paths[idx]
        print(f"\nProcessing: {os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return self._get_default_tensor()

        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        print(f"Frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")

        if total_frames <= 0:
            print("Invalid frame count")
            return self._get_default_tensor()

        # Sample frames uniformly across the video
        frames = []
        frame_indices = []

        # Calculate segment boundaries
        segment_size = total_frames // self.segments_per_video
        for segment_idx in range(self.segments_per_video):
            segment_start = segment_idx * segment_size
            segment_end = segment_start + segment_size

            # Sample frames within segment
            segment_indices = np.linspace(segment_start, segment_end - 1, self.frames_per_segment, dtype=int)
            frame_indices.extend(segment_indices)

        print(f"Sampling {len(frame_indices)} frames")

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx}")
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        if len(frames) < (self.frames_per_segment * self.segments_per_video):
            print(f"Got {len(frames)} frames, padding to required length")
            while len(frames) < (self.frames_per_segment * self.segments_per_video):
                frames.append(frames[-1] if frames else self._get_empty_frame())

        # Shape: [segments_per_video, frames_per_segment, channels, height, width]
        frames_tensor = torch.stack(frames)
        frames_tensor = frames_tensor.view(self.segments_per_video, self.frames_per_segment, 3, 224, 224)

        # Extract label from filename
        combat_score = 1.0 if "combat" in video_path.lower() or "boss" in video_path.lower() else 0.0
        label = torch.tensor([combat_score], dtype=torch.float32)

        print(f"Final tensor shape: {frames_tensor.shape}")
        return frames_tensor, label

    def _get_empty_frame(self):
        return torch.zeros(3, 224, 224, dtype=torch.float32)

    def _get_default_tensor(self):
        frames = torch.zeros(self.segments_per_video, self.frames_per_segment, 3, 224, 224, dtype=torch.float32)
        label = torch.tensor([0.0], dtype=torch.float32)
        return frames, label


class GameCNN(nn.Module):
    def __init__(self, frames_per_segment=16, segments_per_video=10):
        super(GameCNN, self).__init__()
        print("\nInitializing GameCNN")

        # Frame feature extractor (ResNet-18)
        resnet = models.resnet18(pretrained=True)
        self.frame_encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc

        # Temporal encoder for each segment
        self.segment_encoder = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(512, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # Sequence encoder for segments
        self.sequence_encoder = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        print("Network architecture initialized")

    def forward(self, x):
        batch_size, num_segments, frames_per_segment, C, H, W = x.shape
        print(f"\nInput shape: {x.shape}")

        # Process each frame through ResNet
        x = x.view(batch_size * num_segments * frames_per_segment, C, H, W)
        x = self.frame_encoder(x)
        _, C, H, W = x.shape
        print(f"After frame encoding: {x.shape}")

        # Reshape for 3D temporal convolution
        x = x.view(batch_size * num_segments, frames_per_segment, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        print(f"Before temporal conv: {x.shape}")

        # Process each segment
        x = self.segment_encoder(x)
        x = x.view(batch_size, num_segments, -1)
        print(f"After segment encoding: {x.shape}")

        # Process sequence of segments
        x, _ = self.sequence_encoder(x)
        print(f"After sequence encoding: {x.shape}")

        # Global average pooling over segments
        x = torch.mean(x, dim=1)
        print(f"After pooling: {x.shape}")

        # Final classification
        x = self.classifier(x)
        print(f"Final output: {x.shape}")

        return x


def train_model(model, train_loader, val_loader, num_epochs=30, device='cuda'):
    print("\nStarting training...")
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )

    model = model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print(f"\nBatch {batch_idx + 1}/{len(train_loader)}")
            print(f"Input shape: {inputs.shape}")
            print(f"Target shape: {targets.shape}")

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            if batch_idx % 5 == 0:
                print(f'Batch loss: {loss.item():.4f}')

        train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pth')
            print('Saved checkpoint')


def main():
    FRAMES_PER_SEGMENT = 16
    SEGMENTS_PER_VIDEO = 10
    BATCH_SIZE = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    dataset = GameFrameDataset(
        "witcher3_dataset",
        frames_per_segment=FRAMES_PER_SEGMENT,
        segments_per_video=SEGMENTS_PER_VIDEO
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True
    )

    model = GameCNN(
        frames_per_segment=FRAMES_PER_SEGMENT,
        segments_per_video=SEGMENTS_PER_VIDEO
    )

    train_model(model, train_loader, val_loader, device=device)


if __name__ == "__main__":
    main()