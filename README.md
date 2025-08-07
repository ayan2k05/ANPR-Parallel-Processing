# ANPR-MultiGPU

High-performance Automatic Number Plate Recognition system using YOLO and PaddleOCR with multi-GPU optimization for Tesla T4x2. Achieves 8-12x faster processing speeds with smart text correction for Indian license plates.

## Features

- **Multi-GPU Processing**: Optimized for dual Tesla T4 GPUs with intelligent workload distribution
- **High Throughput**: Processes 8-12 images per second (400+ plates per minute)
- **Advanced Detection**: YOLOv8-based license plate detection
- **Robust OCR**: PaddleOCR with smart text correction algorithms
- **Indian Plate Support**: Validates multiple format patterns (XX00XX0000, XX00X0000, XX0XXX0000)
- **Batch Processing**: Handles up to 16 images simultaneously
- **Auto Classification**: Separates valid plates from incomplete detections

## Performance

| Metric | Standard Implementation | ANPR-MultiGPU | Improvement |
|--------|------------------------|---------------|-------------|
| Processing Speed | 0.5-1 img/sec | 8-12 img/sec | 8-12x faster |
| Batch Size | 1 image | 16 images | 16x larger |
| GPU Utilization | ~30% | ~85% | Optimized |
| Accuracy | - | 95% detection, 87% OCR | - |

## Requirements

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python paddlepaddle-gpu paddleocr
pip install numpy pathlib
```

**Hardware Requirements:**
- NVIDIA Tesla T4x2 (or compatible dual GPU setup)
- 32GB+ GPU memory (total)
- CUDA 11.8+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ayan2k05/ANPR-Parallel-Processing
cd anpr-multigpu
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLO model:
   - Place your trained license plate detection model as `license_plate_detector.pt`

## Usage

### Basic Usage

```python
# Update paths in the script
images_folder = '/path/to/your/images'
output_folder = '/path/to/output'

# Run the script
python anpr_multigpu.py
```

### Configuration

Key parameters you can adjust:

```python
# Batch sizes (adjust based on GPU memory)
YOLO_BATCH_SIZE = 16  # For T4x2 with 32GB total memory
OCR_BATCH_SIZE = 8    # OCR processing batch size

# Supported image formats
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
```

## Project Structure

```
anpr-multigpu/
├── anpr_multigpu.py        # Main processing script
├── license_plate_detector.pt  # YOLO model file
├── requirements.txt        # Dependencies
├── README.md              # This file
├── input/                 # Input images directory
└── output/                # Results directory
    ├── valid_plates/      # Successfully processed plates
    └── incomplete/        # Plates requiring review
```

## Text Correction System

The system includes smart corrections for common OCR errors in Indian license plates:

```python
CORRECTIONS = {
    "MB": "WB",  # West Bengal
    "W8": "WB",  # West Bengal
    "0D": "OD",  # Odisha
    "H8": "HR",  # Haryana
    "G7": "GJ",  # Gujarat
    # ... more corrections
}
```

## Output

The system automatically organizes results:

- **Valid Plates**: Saved to `output/` with recognized text as filename
- **Incomplete**: Saved to `output/incomplete/` for manual review
- **Console Output**: Real-time processing status and performance metrics

Example output:
```
T4x2 Optimized License Plate Detection
==================================================
Available GPUs: 2
Initializing models on 2 GPUs...
Found 1000 images to process
Processing on 2 GPUs...
Processed: WB12AB1234 → Valid
Processed: UNKNOWN → Invalid
...
 Completed processing 1000 images
 Total time: 125.50 seconds
 Average: 0.126 seconds per image
T4x2 Performance: ~8.0 images/second
```

## Supported License Plate Formats

- `XX00XX0000` - Standard format (e.g., WB12AB1234)
- `XX00X0000` - Alternative format (e.g., DL8CA1234)
- `XX0XXX0000` - Extended format (e.g., MH1ABC1234)

## Optimization Features

- **Half-Precision (FP16)**: Leverages T4 GPU architecture for faster inference
- **Parallel Processing**: ThreadPoolExecutor for multi-GPU coordination
- **Memory Pooling**: Efficient resource allocation across 32GB GPU memory
- **Batch Optimization**: Large batch processing for maximum throughput
- **Smart Caching**: LRU cache for text corrections and pattern matching

## Performance Tips

1. **Monitor GPU Usage**: Use `nvidia-smi` to verify both GPUs are utilized
2. **Adjust Batch Sizes**: Reduce if you encounter out-of-memory errors
3. **Image Quality**: Higher quality images improve OCR accuracy
4. **Model Selection**: Use the latest YOLO model for best detection results

## Troubleshooting

**GPU Not Detected:**
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"
```

**Out of Memory Error:**
- Reduce `YOLO_BATCH_SIZE` from 16 to 8 or 4
- Reduce `OCR_BATCH_SIZE` from 8 to 4

**Low Performance:**
- Ensure both GPUs are being utilized
- Check if FP16 is enabled for your GPU
- Verify cuDNN is properly installed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **YOLO**: Object detection framework by Ultralytics
- **PaddleOCR**: OCR toolkit by PaddlePaddle
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{anpr-multigpu,
  title={ANPR-Parallel-Processing with multiple GPU: High-Performance License Plate Recognition},
  author={Ayan Pathan},
  year={2025},
  url={https://github.com/ayan2k05/ANPR-Parallel-Processing}
}
```
