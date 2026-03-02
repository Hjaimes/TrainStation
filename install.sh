#!/usr/bin/env bash
# ============================================================
# TrainStation — Linux/Mac Installer
# Creates a venv, installs PyTorch with CUDA, then all deps.
# ============================================================
set -e

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  TrainStation — Dependency Installer${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

cd "$(dirname "$0")"

# ============================================================
# 1. Find Python
# ============================================================
echo -e "${CYAN}[1/6]${NC} Locating Python..."

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PY_VER=$("$cmd" --version 2>&1)
        echo "  Found: $PY_VER"
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}ERROR: Python not found.${NC}"
    echo "  Install Python 3.10+ from your package manager or https://www.python.org/"
    exit 1
fi

# Verify version is 3.10+
if ! "$PYTHON" -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo -e "${RED}ERROR: Python 3.10 or newer is required. Found: $PY_VER${NC}"
    exit 1
fi
echo -e "  ${GREEN}OK${NC}"
echo ""

# ============================================================
# 2. Create or reuse venv
# ============================================================
echo -e "${CYAN}[2/6]${NC} Setting up virtual environment..."

if [ -f "venv/bin/activate" ]; then
    echo "  Existing venv found — reusing it."
else
    echo "  Creating new venv..."
    "$PYTHON" -m venv venv
    echo -e "  ${GREEN}Created.${NC}"
fi

source venv/bin/activate
echo -e "  ${GREEN}Activated.${NC}"
echo ""

# ============================================================
# 3. Upgrade pip
# ============================================================
echo -e "${CYAN}[3/6]${NC} Upgrading pip..."
python -m pip install --upgrade pip --quiet
echo -e "  ${GREEN}Done.${NC}"
echo ""

# ============================================================
# 4. Choose CUDA or CPU
# ============================================================
echo -e "${CYAN}[4/6]${NC} PyTorch installation"
echo ""
echo "  Which PyTorch variant do you want to install?"
echo ""
echo "    1) CUDA 12.8  (NVIDIA GPU — recommended)"
echo "    2) CUDA 12.4  (NVIDIA GPU — older drivers)"
echo "    3) CPU only   (no GPU acceleration)"
echo "    4) Skip       (PyTorch already installed)"
echo ""
read -p "  Enter choice [1-4] (default: 1): " TORCH_CHOICE
TORCH_CHOICE=${TORCH_CHOICE:-1}

case "$TORCH_CHOICE" in
    1)
        echo ""
        echo "  Installing PyTorch with CUDA 12.8..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
        ;;
    2)
        echo ""
        echo "  Installing PyTorch with CUDA 12.4..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
        ;;
    3)
        echo ""
        echo "  Installing PyTorch (CPU only)..."
        pip install torch torchvision
        ;;
    4)
        echo ""
        echo "  Skipping PyTorch installation."
        if ! python -c "import torch; print(f'  Found torch {torch.__version__}')" 2>/dev/null; then
            echo -e "  ${YELLOW}WARNING: torch not found — you may need to install it manually.${NC}"
        fi
        ;;
    *)
        echo -e "${RED}Invalid choice. Defaulting to CUDA 12.8.${NC}"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
        ;;
esac

echo -e "  ${GREEN}Done.${NC}"
echo ""

# ============================================================
# 5. Install base requirements
# ============================================================
echo -e "${CYAN}[5/6]${NC} Installing dependencies..."
pip install -r requirements/base.txt
echo -e "  ${GREEN}Base dependencies installed.${NC}"

# Optional optimizers
echo ""
read -p "  Install optional optimizers (bitsandbytes, prodigy, lion, came, schedulefree)? [y/N]: " OPT_CHOICE
if [[ "$OPT_CHOICE" =~ ^[Yy]$ ]]; then
    echo "  Installing optional optimizers..."
    pip install -r requirements/optimizers.txt || {
        echo -e "  ${YELLOW}WARNING: Some optional optimizers failed to install. This is OK.${NC}"
    }
    echo -e "  ${GREEN}Optional optimizers installed.${NC}"
fi
echo ""

# ============================================================
# 6. Build frontend
# ============================================================
echo -e "${CYAN}[6/6]${NC} Building frontend..."

if command -v npm &>/dev/null; then
    cd ui/frontend
    npm install --silent 2>/dev/null
    npm run build || {
        echo -e "  ${YELLOW}WARNING: Frontend build failed. The server will still work${NC}"
        echo -e "  ${YELLOW}but the UI won't load. Try: cd ui/frontend && npm install && npm run build${NC}"
    }
    cd ../..
    echo -e "  ${GREEN}Frontend built.${NC}"
else
    echo -e "  ${YELLOW}WARNING: npm not found — skipping frontend build.${NC}"
    echo "  Install Node.js from https://nodejs.org/ then run:"
    echo "    cd ui/frontend && npm install && npm run build"
fi
echo ""

# ============================================================
# Done
# ============================================================
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Installation complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  To start the UI:"
echo -e "    ${CYAN}./start_ui.sh${NC}"
echo ""
echo "  Or manually:"
echo -e "    ${CYAN}source venv/bin/activate${NC}"
echo -e "    ${CYAN}python run_ui.py${NC}"
echo ""
echo "  To train from CLI:"
echo -e "    ${CYAN}source venv/bin/activate${NC}"
echo -e "    ${CYAN}python run.py --config your_config.yaml${NC}"
echo ""
