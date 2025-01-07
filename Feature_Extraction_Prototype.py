# Feature_Extraction_Prototype.py
import torch
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import torch.nn as nn
import torchvision.models as models
import os

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

        # Efficient classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
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
            frame_features.append(features.cpu())
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


class Witcher3Analyzer:
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Initialize model
        self.model = MemoryEfficientGameCNN(
            frames_per_segment=16,
            segments_per_video=10,
            device=device
        )

        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.frames_per_segment = 16
        self.segments_per_video = 10

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def analyze_video(self, video_path: str) -> Dict:
        """Analyze a Witcher 3 gameplay video and return combat timestamps."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"\nAnalyzing video: {video_path}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps}")

        # Store combat probabilities and their timestamps
        combat_data = []

        # Process video in segments
        with torch.no_grad():
            for start_frame in tqdm(range(0, total_frames, self.frames_per_segment * self.segments_per_video)):
                frames = []
                for i in range(self.segments_per_video):
                    segment_frames = self._extract_frames(cap, start_frame + i * self.frames_per_segment,
                                                          self.frames_per_segment)
                    if not segment_frames:
                        continue
                    frames.extend(segment_frames)

                if len(frames) < self.frames_per_segment * self.segments_per_video:
                    continue

                # Stack frames and reshape for model
                frames_tensor = torch.stack(frames)
                frames_tensor = frames_tensor.view(1, self.segments_per_video, self.frames_per_segment, 3, 224, 224)

                # Get combat probability for this segment
                combat_prob = self._predict_segment(frames_tensor)
                timestamp = start_frame / fps

                combat_data.append({
                    'timestamp': timestamp,
                    'combat_probability': float(combat_prob),
                    'is_combat': combat_prob > 0.5
                })

        cap.release()

        # Analyze the combat data
        analysis = self._analyze_combat_data(combat_data, fps)

        return {
            'video_info': {
                'path': video_path,
                'duration': duration,
                'total_frames': total_frames,
                'fps': fps
            },
            'combat_data': combat_data,
            'analysis': analysis
        }

    def _extract_frames(self, cap, start_frame: int, num_frames: int) -> List[torch.Tensor]:
        """Extract and preprocess frames from the video."""
        frames = []
        for frame_idx in range(start_frame, start_frame + num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            frames.append(frame)

        return frames

    def _predict_segment(self, frames_tensor: torch.Tensor) -> float:
        """Predict combat probability for a segment of frames."""
        frames_tensor = frames_tensor.to(self.device)

        # Get model prediction
        with torch.cuda.amp.autocast():
            output = self.model(frames_tensor)
            prob = torch.sigmoid(output).cpu().item()

        return prob

    def _analyze_combat_data(self, combat_data: List[Dict], fps: float) -> Dict:
        """Analyze combat segments to extract meaningful statistics."""
        combat_segments = []
        current_segment = None

        # Identify combat segments
        for entry in combat_data:
            if entry['is_combat'] and current_segment is None:
                current_segment = {'start': entry['timestamp']}
            elif not entry['is_combat'] and current_segment is not None:
                current_segment['end'] = entry['timestamp']
                current_segment['duration'] = current_segment['end'] - current_segment['start']
                combat_segments.append(current_segment)
                current_segment = None

        # Calculate statistics
        total_combat_time = sum(seg['duration'] for seg in combat_segments) if combat_segments else 0
        avg_combat_duration = total_combat_time / len(combat_segments) if combat_segments else 0

        return {
            'total_combat_segments': len(combat_segments),
            'total_combat_time': total_combat_time,
            'average_combat_duration': avg_combat_duration,
            'combat_segments': combat_segments
        }

    def visualize_combat_timeline(self, analysis_result: Dict, save_path: str = None):
        """Create a visualization of combat occurrences throughout the video."""
        timestamps = [d['timestamp'] for d in analysis_result['combat_data']]
        probabilities = [d['combat_probability'] for d in analysis_result['combat_data']]

        plt.figure(figsize=(15, 5))
        plt.plot(timestamps, probabilities, 'b-', label='Combat Probability')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Combat Threshold')

        # Highlight combat segments
        for segment in analysis_result['analysis']['combat_segments']:
            plt.axvspan(segment['start'], segment['end'], alpha=0.2, color='red')

        plt.xlabel('Time (seconds)')
        plt.ylabel('Combat Probability')
        plt.title('Combat Detection Timeline')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class FeatureVisualizer:
    def __init__(self, model_path: str, output_dir: str = 'feature_analysis'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load model same as before
        self.model = MemoryEfficientGameCNN(
            frames_per_segment=16,
            segments_per_video=10,
            device=self.device
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, video_path: str, save_features: bool = True):
        """Extract and visualize features from key moments in the video."""
        print(f"\nExtracting features from: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Storage for interesting moments and features
        feature_data = []
        all_features = []

        # Process video in segments
        with torch.no_grad():
            for frame_idx in range(0, total_frames, 16):  # Process every 16th frame
                # Read and transform frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Save original frame
                original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Get model features
                frame_tensor = self.transform(original_frame).unsqueeze(0).to(self.device)

                # Extract intermediate features
                spatial_features = self.model.frame_encoder(frame_tensor)
                
                # Store the features
                all_features.append(spatial_features.cpu())

                # Convert features to heatmap for visualization
                feature_maps = spatial_features.mean(dim=1).cpu().numpy()[0]
                heatmap = self._create_heatmap(feature_maps, original_frame.shape[:2])

                # Combine original frame and heatmap
                visualization = self._create_visualization(original_frame, heatmap)

                # Save if it's an interesting moment (high activation)
                if np.mean(heatmap) > 0.5:  # Threshold for "interesting" features
                    timestamp = frame_idx / fps
                    save_path = os.path.join(self.output_dir, f'feature_frame_{frame_idx}.png')

                    if save_features:
                        plt.imsave(save_path, visualization)

                    feature_data.append({
                        'frame_idx': frame_idx,
                        'timestamp': timestamp,
                        'feature_intensity': float(np.mean(heatmap)),
                        'save_path': save_path
                    })

                # Clean up
                del spatial_features
                torch.cuda.empty_cache()

        cap.release()
        
        # Stack all features together
        if all_features:
            stacked_features = torch.cat(all_features, dim=0)
        else:
            stacked_features = torch.zeros((1, 512, 7, 7))  # Default size for ResNet18 features
        
        return {
            'features': stacked_features,
            'feature_data': feature_data
        }

    def _create_heatmap(self, feature_maps, target_size):
        """Convert feature maps to a heatmap."""
        # Normalize feature maps
        heatmap = np.mean(np.abs(feature_maps), axis=0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Resize to match original frame
        heatmap = cv2.resize(heatmap, (target_size[1], target_size[0]))
        return heatmap

    def _create_visualization(self, original_frame, heatmap):
        """Create a side-by-side visualization of frame and features."""
        # Convert heatmap to color
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Create overlay
        overlay = (0.7 * original_frame + 0.3 * heatmap_colored).astype(np.uint8)

        # Stack original and overlay side by side
        return np.hstack((original_frame, overlay))


def main():
    # Initialize feature visualizer
    visualizer = FeatureVisualizer(
        model_path='training_checkpoints/best_model.pth',
        output_dir='feature_analysis'
    )

    # Extract features from video
    video_path = "witcher3_dataset/Caranthir Boss Fight (No Damage⧸NG+⧸DM) The Witcher 3.f298.mp4"
    feature_data = visualizer.extract_features(video_path)

    # Print analysis
    print("\nFeature Analysis Results:")
    print(f"Total interesting moments found: {len(feature_data)}")
    for moment in feature_data:
        print(f"\nTimestamp: {moment['timestamp']:.2f}s")
        print(f"Feature intensity: {moment['feature_intensity']:.3f}")
        print(f"Saved to: {moment['save_path']}")


if __name__ == "__main__":
    main()