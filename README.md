# TrainStation

A modern AI model training application with a web UI and CLI. Fine-tune diffusion and video generation models with LoRA, LoHa, LoKr, and full fine-tuning.

> **Note:** TrainStation is under active development. Some features may be incomplete or experimental.

## Supported Architectures

| Architecture | Type | Status |
|---|---|---|
| Wan (T2V, I2V, FLF2V) | Video | Fully implemented |
| HunyuanVideo | Video | Fully implemented |
| HunyuanVideo 1.5 | Video | Fully implemented |
| FramePack | Video | Fully implemented |
| Flux 1 | Image | Fully implemented |
| Flux 2 | Image | Fully implemented |
| Flux Kontext | Image | Fully implemented |
| SDXL | Image | Fully implemented |
| SD3 | Image | Fully implemented |
| Kandinsky 5 | Image | Fully implemented |
| Qwen Image | Image | Fully implemented |
| Z-Image | Image | Fully implemented |

## Features

**Training Methods**
- LoRA, LoHa, LoKr, and full fine-tuning
- Multiple loss functions (MSE, L1, Huber)
- Min-SNR gamma and debiased loss weighting
- Configurable timestep sampling (uniform, sigmoid, logit-normal)
- EMA (Exponential Moving Average) tracking

**Optimizers & Schedulers**
- AdamW, AdamW8bit, Adafactor, Prodigy, LION, CAME, Schedule-Free AdamW
- Cosine, constant, linear, warmup, exponential, inverse sqrt schedulers

**VRAM Optimization**
- Block swap (offload transformer blocks to CPU)
- FP8 / NF4 / INT8 quantization
- Gradient checkpointing

**Data Pipeline**
- Latent and text encoder output caching
- Resolution bucketing with configurable min/max
- Horizontal flip augmentation

**UI & Workflow**
- Web UI with live loss chart and real-time training metrics
- HuggingFace model IDs - type a repo ID (e.g. `Wan-AI/Wan2.1-T2V-14B`) and it downloads automatically
- Native file/folder browser
- Preset system - save and load training configurations
- Training queue - queue multiple runs
- CLI for headless/scripted training

## Requirements

- **Python** 3.10 or newer
- **Node.js** 18+ (for building the frontend)
- **NVIDIA GPU** with CUDA support (recommended)
  - 8 GB VRAM: LoRA training with quantization + block swap
  - 12 GB VRAM: LoRA training with quantization
  - 24 GB+ VRAM: Full fine-tuning (smaller models) or comfortable LoRA training

> **CPU-only** is technically supported but not practical for training. It can be useful for testing configs.

## Installation

### Quick Start (Windows)

```
git clone https://github.com/Hjaimes/TrainStation.git
cd TrainStation
install.bat
```

### Quick Start (Linux / Mac)

```bash
git clone https://github.com/Hjaimes/TrainStation.git
cd TrainStation
chmod +x install.sh start_ui.sh start_train.sh update.sh
./install.sh
```

The installer will:
1. Create a Python virtual environment
2. Ask which PyTorch version to install (CUDA 12.8, 12.4, or CPU)
3. Install all dependencies
4. Optionally install extra optimizers (bitsandbytes, Prodigy, LION, CAME, Schedule-Free)
5. Build the frontend

## Usage

### Web UI

```
start_ui.bat          # Windows
./start_ui.sh         # Linux / Mac
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

### CLI

```
start_train.bat --config your_config.yaml          # Windows
./start_train.sh --config your_config.yaml          # Linux / Mac
```

### Updating

```
update.bat            # Windows
./update.sh           # Linux / Mac
```

Pulls the latest code from GitHub, updates dependencies, and rebuilds the frontend.

## Troubleshooting

**"Virtual environment not found"**
Run `install.bat` (Windows) or `./install.sh` (Linux/Mac) first.

**Frontend not loading / blank page**
Make sure Node.js is installed, then rebuild:
```bash
cd ui/frontend && npm install && npm run build
```

**CUDA out of memory**
- Enable quantization (FP8 or NF4) in the Model tab
- Increase the block swap count
- Reduce batch size to 1
- Enable gradient checkpointing (on by default)

**Optional optimizers not available**
Install them manually:
```bash
pip install bitsandbytes prodigyopt lion-pytorch came-pytorch schedulefree
```

## Project Structure

```
trainer/              # Training backend (pure Python, no UI dependencies)
  arch/               # Model architectures (one package per arch)
  config/             # Pydantic v2 config schema, validation, I/O
  data/               # Dataset loading, caching, augmentations
  networks/           # LoRA, LoHa, LoKr, DoRA modules
  training/           # Trainer loop, methods, session orchestration
  util/               # Utilities (dtype, timer, HF hub, etc.)
ui/                   # Web UI backend (FastAPI + WebSocket)
  frontend/           # SvelteKit frontend
  routes/             # API endpoints
tests/                # Test suite
```

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run tests before submitting: `python -m pytest tests/ -v`
4. Open a pull request

Please keep changes focused and include tests for new functionality.

## Acknowledgements

TrainStation builds on the work of several open-source projects:

- [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner) by kohya-ss - primary reference for model loading, LoRA/LoHa/LoKr networks, dataset handling, and training loops (Apache 2.0)
- [OneTrainer](https://github.com/Nerogar/OneTrainer) by Nerogar - reference for preset systems, factory patterns, and UI state management (AGPL-3.0)
- [AI Toolkit](https://github.com/ostris/ai-toolkit) by ostris - reference for YAML config approach and job-based architecture (MIT)

## License

[Apache License 2.0](LICENSE)
