#!/usr/bin/env bash
# ============================================================
# TrainStation — Linux/Mac Updater
# Pulls latest from GitHub, reinstalls deps, rebuilds frontend.
# ============================================================
set -e

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  TrainStation — Updater${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

cd "$(dirname "$0")"

# ============================================================
# 1. Check prerequisites
# ============================================================
echo -e "${CYAN}[1/4]${NC} Checking prerequisites..."

if ! command -v git &>/dev/null; then
    echo -e "${RED}ERROR: git not found. Install Git first.${NC}"
    exit 1
fi

if [ ! -f "venv/bin/activate" ]; then
    echo -e "${RED}ERROR: Virtual environment not found. Run install.sh first.${NC}"
    exit 1
fi

echo -e "  ${GREEN}OK${NC}"
echo ""

# ============================================================
# 2. Pull latest changes
# ============================================================
echo -e "${CYAN}[2/4]${NC} Pulling latest changes from GitHub..."

STASHED=0
if ! git diff --quiet 2>/dev/null; then
    echo -e "  ${YELLOW}WARNING: You have local changes. Stashing them...${NC}"
    git stash
    STASHED=1
fi

if ! git pull --ff-only 2>/dev/null; then
    echo -e "  ${YELLOW}Fast-forward failed. Trying rebase...${NC}"
    if ! git pull --rebase; then
        echo -e "${RED}ERROR: Could not pull latest changes. You may have conflicting local modifications.${NC}"
        if [ "$STASHED" -eq 1 ]; then
            echo "  Restoring your stashed changes..."
            git stash pop || true
        fi
        exit 1
    fi
fi

if [ "$STASHED" -eq 1 ]; then
    echo "  Restoring your local changes..."
    git stash pop || {
        echo -e "  ${YELLOW}WARNING: Could not auto-restore stashed changes. Run 'git stash pop' manually.${NC}"
    }
fi

echo -e "  ${GREEN}Done.${NC}"
echo ""

# ============================================================
# 3. Update Python dependencies
# ============================================================
echo -e "${CYAN}[3/4]${NC} Updating Python dependencies..."

source venv/bin/activate
pip install -r requirements/base.txt --quiet
echo -e "  ${GREEN}Done.${NC}"
echo ""

# ============================================================
# 4. Rebuild frontend
# ============================================================
echo -e "${CYAN}[4/4]${NC} Rebuilding frontend..."

if command -v npm &>/dev/null; then
    cd ui/frontend
    npm install --silent 2>/dev/null
    npm run build || {
        echo -e "  ${YELLOW}WARNING: Frontend build failed. Try: cd ui/frontend && npm install && npm run build${NC}"
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
echo -e "${GREEN}  Update complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "  Start the UI with: ${CYAN}./start_ui.sh${NC}"
echo ""
