# data_handling.py
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Dict, Tuple
import logging
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import gc
import time
from datetime import datetime

# local imports (unchanged as requested)
from Feature_Extraction_Prototype import Witcher3Analyzer, FeatureVisualizer


def log_time(message: str):
    """Helper function to log timestamps"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


class FeatureDataset(Dataset):
    def __init__(self, feature_dir: str, cache_size: int = 100):
        """Dataset for loading pre-extracted features with caching"""
        log_time(f"Initializing FeatureDataset with directory: {feature_dir}")

        if not os.path.exists(feature_dir):
            raise FileNotFoundError(f"Directory not found: {feature_dir}")

        self.feature_files = [
            os.path.join(feature_dir, f) for f in os.listdir(feature_dir)
            if f.endswith('.pt')
        ]
        log_time(f"Found {len(self.feature_files)} feature files")

        if not self.feature_files:
            raise ValueError(f"No feature files found in {feature_dir}")

        # Load first file to get dimensions
        log_time("Loading first file to get dimensions...")
        sample = torch.load(self.feature_files[0])
        self.feature_size = sample['features'].shape[-1]

        # Initialize cache
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

        print(f"\nFeature Dataset initialized with:")
        print(f"Number of sequences: {len(self.feature_files)}")
        print(f"Feature size: {self.feature_size}")
        print(f"Cache size: {cache_size}")
        log_time("FeatureDataset initialization complete")

    def __len__(self) -> int:
        return len(self.feature_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        start_time = time.time()
        try:
            # Check cache first
            if idx in self.cache:
                self.cache_hits += 1
                if self.cache_hits % 100 == 0:
                    log_time(f"Cache hits: {self.cache_hits}, Cache misses: {self.cache_misses}")
                return self.cache[idx]

            self.cache_misses += 1
            log_time(f"Loading file: {self.feature_files[idx]}")

            data = torch.load(self.feature_files[idx])
            result = (data['features'], {
                'combat': data['combat_label'],
                'resource': data['resource_label'],
                'quest': data['quest_label']
            })

            # Update cache
            if len(self.cache) >= self.cache_size:
                log_time("Cache full, removing oldest item")
                self.cache.pop(next(iter(self.cache)))
            self.cache[idx] = result

            load_time = time.time() - start_time
            if load_time > 1.0:  # Log if loading takes more than 1 second
                log_time(f"Slow file load: {self.feature_files[idx]}, took {load_time:.2f}s")

            return result
        except Exception as e:
            log_time(f"Error loading file {self.feature_files[idx]}: {str(e)}")
            return self._get_default_item()

    def _get_default_item(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        log_time("Returning default item due to error")
        features = torch.zeros(30, self.feature_size)
        labels = {
            'combat': torch.zeros(3),
            'resource': torch.zeros(4),
            'quest': torch.zeros(2)
        }
        return features, labels


def process_video_batch(args):
    """Helper function to process a single video"""
    video_path, feature_path, visualizer = args
    start_time = time.time()
    log_time(f"Starting processing of {os.path.basename(video_path)}")

    try:
        if os.path.exists(feature_path):
            log_time(f"Skipping existing file: {os.path.basename(video_path)}")
            return f"Skipped existing {os.path.basename(video_path)}"

        log_time(f"Extracting features from {os.path.basename(video_path)}")
        result = visualizer.extract_features(video_path, save_features=False)
        log_time(f"Feature extraction completed for {os.path.basename(video_path)}")

        # Calculate average combat probability
        feature_intensities = [d['feature_intensity'] for d in result['feature_data']]
        avg_combat_prob = np.mean(feature_intensities) if feature_intensities else 0.5

        features_dict = {
            'features': result['features'],
            'combat_label': torch.tensor([
                avg_combat_prob,
                avg_combat_prob,
                avg_combat_prob
            ], dtype=torch.float32),
            'resource_label': torch.tensor([
                1.0 - 0.3 * avg_combat_prob,
                1.0 - 0.4 * avg_combat_prob,
                0.5 + 0.5 * avg_combat_prob,
                0.7
            ], dtype=torch.float32),
            'quest_label': torch.tensor([0.5, 0.3], dtype=torch.float32)
        }

        log_time(f"Saving features for {os.path.basename(video_path)}")
        torch.save(features_dict, feature_path)

        # Clear some memory
        del result, features_dict
        gc.collect()

        process_time = time.time() - start_time
        log_time(f"Completed processing {os.path.basename(video_path)} in {process_time:.2f}s")
        return f"Processed {os.path.basename(video_path)} in {process_time:.2f}s"

    except Exception as e:
        log_time(f"Error processing {os.path.basename(video_path)}: {str(e)}")
        return f"Error processing {os.path.basename(video_path)}: {str(e)}"


def extract_and_save_features(
        video_dir: str,
        output_dir: str,
        checkpoint_path: str,
        max_workers: int = 2
):
    """Extract features from videos using the trained CNN model with parallel processing"""
    start_time = time.time()
    log_time(f"Starting feature extraction with {max_workers} workers")

    os.makedirs(output_dir, exist_ok=True)
    video_dir = os.path.normpath(video_dir)

    log_time("Initializing FeatureVisualizer...")
    visualizer = FeatureVisualizer(checkpoint_path, output_dir)
    log_time("FeatureVisualizer initialized")

    log_time(f"Scanning directory: {video_dir}")
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    log_time(f"Found {len(video_files)} video files")

    # Prepare arguments for parallel processing
    process_args = [
        (
            os.path.join(video_dir, video_name),
            os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_features.pt"),
            visualizer
        )
        for video_name in video_files
    ]

    log_time(f"Starting parallel processing with {max_workers} workers")
    # Process videos in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_video_batch, process_args),
            total=len(video_files),
            desc="Extracting features"
        ))

    # Print results
    for result in results:
        log_time(result)

    total_time = time.time() - start_time
    log_time(f"Total processing time: {total_time:.2f}s")


def create_feature_datasets(
        train_dir: str,
        val_dir: str,
        batch_size: int,
        num_workers: int = 0,
        cache_size: int = 100
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders with caching"""
    log_time("Creating feature datasets...")
    log_time(f"Training directory: {train_dir}")
    log_time(f"Validation directory: {val_dir}")
    log_time(f"Batch size: {batch_size}")
    log_time(f"Number of workers: {num_workers}")

    train_dataset = FeatureDataset(train_dir, cache_size=cache_size)
    val_dataset = FeatureDataset(val_dir, cache_size=cache_size)

    log_time("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    log_time("Data loaders created successfully")

    return train_loader, val_loader


if __name__ == "__main__":
    log_time("Script started")

    # Set device optimization
    if torch.cuda.is_available():
        log_time("CUDA is available, setting optimization flags")
        torch.backends.cudnn.benchmark = True
    else:
        log_time("CUDA is not available, running on CPU")

    # Extract features with parallel processing
    extract_and_save_features(
        video_dir="witcher3_dataset",
        output_dir="extracted_features",
        checkpoint_path='training_checkpoints/best_model.pth',
        max_workers=2  # Adjust based on your CPU cores
    )

    # Create dataloaders with optimized settings
    log_time("Creating dataloaders...")
    train_loader, val_loader = create_feature_datasets(
        train_dir="train_features",
        val_dir="val_features",
        batch_size=32,
        num_workers=2,
        cache_size=100
    )

    log_time("Script completed")