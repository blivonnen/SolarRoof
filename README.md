# SolarRoof

A deep learning project for solar roof segmentation using U-Net architecture. This project trains a neural network to identify and segment rooftops from satellite imagery.

## Overview

SolarRoof uses a U-Net convolutional neural network to perform semantic segmentation of rooftop from satellite images. The model is trained on a dataset of satellite images with corresponding segmentation masks, learning to identify rooftop locations and boundaries.

## Features

- **U-Net Architecture**: Lightweight convolutional neural network optimized for image segmentation
- **Mixed Precision Training**: Support for FP16 training on NVIDIA Tensor Core GPUs
- **Hardware Optimization**: Automatic detection and optimization for different GPU types (L4, A100, etc.)
- **Cloud Deployment**: Ready for deployment on OVHcloud AI Training platform
- **Data Pipeline**: Efficient data loading from HuggingFace datasets using Apache Arrow/Parquet

## Project Structure

```
SolarRoof/
├── main.py              # Training script with CLI arguments
├── infer.py             # Inference and visualization script
├── visualize.py         # Data exploration and visualization
├── Dockerfile           # Container configuration for cloud deployment
├── .requirements.txt    # Python dependencies
├── model-output/        # Trained model files
│   ├── solar_roof_unet_v1.h5
│   ├── solar_roof_unet.h5
│   └── solar_roof_unet.keras
└── .github/workflows/   # CI/CD pipeline for S3 sync
    └── s3-sync.yml
```

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd SolarRoof
```

2. Install dependencies:
```bash
pip install -r .requirements.txt
```

### Docker Deployment

Build the Docker image:
```bash
docker build -t solar-roof .
```

## Usage

### Training

Train the model with default settings:
```bash
python main.py
```

Customize training parameters:
```bash
python main.py --gpu a100 --epochs 50 --modeloutput ./model-output/solar_roof_unet_v2.h5
```

#### Command Line Arguments

- `--gpu`: GPU type (`auto`, `cpu`, `l4`, `a100`, `ada`, `ampere`)
- `--epochs`: Number of training epochs (default: 30)
- `--modeloutput`: Path for saving the trained model

### Inference

Run inference on test images:
```bash
python infer.py
```

This will:
1. Load the trained model from `model-output/solar_roof_unet.h5`
2. Sample random images from the dataset
3. Generate predictions and display original, conditioning, and predicted images

### Data Visualization

Explore the dataset:
```bash
python visualize.py
```

## Model Architecture

The project implements a lightweight U-Net with the following characteristics:

- **Input**: 256×256 RGB images
- **Output**: 256×256 grayscale segmentation masks
- **Encoder**: 3 downsampling blocks with increasing feature maps (32→64→128)
- **Bottleneck**: 256 feature maps
- **Decoder**: 3 upsampling blocks with skip connections
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam

## Hardware Optimization

The training script automatically detects and optimizes for different hardware:

- **CPU**: Forces CPU-only mode
- **NVIDIA Tensor Core GPUs** (L4, A100, Ada, Ampere): Enables mixed precision (FP16) and XLA JIT compilation
- **Other GPUs**: Standard training mode

## Cloud Deployment

### OVHcloud AI Training

The project is configured for deployment on OVHcloud AI Training platform:

1. **Build and push the Docker image**:
```bash
docker build -t blivonnen/solar-roof:latest .
docker push blivonnen/solar-roof:latest
```

2. **Run training job**:
```bash
ovhai job run --name solar-roof-job \
  --flavor ai1-le-1-gpu \
  -v blivonnen-s3@my-os:/workspace:rw \
  blivonnen/solar-roof:latest \
  -- bash -c 'python3 /workspace/SolarRoof/main.py --epochs 30'
```

### S3 Integration

The project includes GitHub Actions workflow for automatic S3 synchronization:

- Automatically syncs code changes to OVH S3 bucket
- Triggers on pushes to master branch
- Excludes hidden files and directories

## Dataset

The model is trained on the roof segmentation dataset from HuggingFace:
- **Source**: `hf://datasets/dpanangian/roof-segmentation-control-net/data/train-00000-of-00001.parquet`
- **Format**: Apache Arrow/Parquet with embedded image bytes
- **Content**: Satellite images with corresponding segmentation masks

## Dependencies

Core dependencies include:
- `tensorflow>=2.15.0` - Deep learning framework
- `scikit-learn` - Data splitting and utilities
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `Pillow` - Image processing
- `fsspec` - File system interface
- `huggingface_hub` - Dataset access
- `pyarrow>=14` - Parquet file handling

## Performance

- **Training Time**: ~30 minutes on A100 GPU (30 epochs)
- **Memory Usage**: ~8GB GPU memory with batch size 16
- **Model Size**: ~15MB (lightweight U-Net)
- **Inference Speed**: ~50ms per image on GPU

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- Dataset provided by HuggingFace
- U-Net architecture based on the original paper by Ronneberger et al.
- Cloud deployment optimized for OVHcloud AI Training platform 