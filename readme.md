# Witcher 3 Gameplay Analysis System

A deep learning pipeline for automated analysis of Witcher 3 gameplay footage using CNN-LSTM architecture. The system analyzes combat efficiency, resource management, and quest progression in real-time.

## 🎯 Features

- **Data Collection**: YouTube gameplay footage scraping with filters for specific content types
- **Frame Analysis**: CNN-based feature extraction for gameplay frames
- **Temporal Analysis**: LSTM/Transformer for sequence understanding
- **Multi-task Learning**: Simultaneous analysis of multiple gameplay aspects
- **Memory-Efficient Processing**: Optimized for processing long gameplay videos

## 🔧 System Components

### 1. Data Collection (`Witcher_Data_Scrapper.py`)
- YouTube gameplay footage scraping system
- Configurable search queries for different gameplay types
- Automatic categorization of downloaded content
- Built-in rate limiting and error handling

### 2. Feature Extraction (`CNN_Feature_Extraction.py` & `Feature_Extraction_Prototype.py`)
- ResNet18-based frame feature extraction
- Memory-efficient processing with gradient checkpointing
- Temporal feature aggregation
- Real-time feature visualization capabilities

### 3. Data Handling (`data_handling.py`)
- Custom PyTorch datasets for video processing
- Efficient caching mechanism for faster training
- Parallel processing support
- Robust error handling and recovery

### 4. CNN Model (`CNN_Model.py`)
- Memory-efficient architecture combining ResNet and LSTM
- Multi-task output heads
- Checkpoint system for training recovery
- Mixed precision training support

## 📊 Model Architecture

```
Input Video -> CNN Frame Analysis -> Temporal Processing -> Multi-task Outputs
                    ↓                        ↓                     ↓
              ResNet Features     →    LSTM/Transformer   →    Combat Score
                                                        →    Resource Rating
                                                        →    Quest Progress
```

## ⚙️ Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA-capable GPU
- Required packages in `requirements.txt`

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download training data:
```bash
python Witcher_Data_Scrapper.py
```

3. Train the model:
```bash
python CNN_Model.py
```

4. Run analysis on a video:
```bash
python Feature_Extraction_Prototype.py --video path/to/video.mp4
```

## 📝 Project Structure

```
witcher3-gameplay-ai-analysis-ml-cv/
├── Witcher_Data_Scrapper.py    # YouTube data collection
├── Feature_Extraction_Prototype.py  # Main feature extraction
├── CNN_Feature_Extraction.py    # CNN architecture
├── CNN_Model.py                # Training pipeline
├── data_handling.py            # Dataset management
└── training_checkpoints/       # Model checkpoints
```

## 🔄 Current Status

- [x] CNN Frame Analysis
- [ ] LSTM/Transformer Integration (In Progress)
- [ ] Multi-task Learning Heads
- [ ] Real-time Analysis System

## 📈 Performance Notes

The system is designed for memory efficiency, handling long gameplay videos through:
- Gradient checkpointing
- Batch processing
- Mixed precision training
- Efficient caching mechanisms

## 🛠️ Development Notes

- The CNN model achieves ~96% accuracy on combat detection
- Current focus on LSTM/Transformer integration
- Future plans include real-time analysis capabilities
- Memory optimization is a key priority


## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details