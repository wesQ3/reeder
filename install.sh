#!/bin/bash
set -euo pipefail

# Reeder Installation Script
# Installs dependencies and sets up systemd services

INSTALL_DIR="${REEDER_INSTALL_DIR:-/var/lib/reeder}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Reeder Installation ==="
echo "Install directory: $INSTALL_DIR"
echo ""

# Detect package manager
if command -v pacman &> /dev/null; then
    PKG_MGR="pacman"
    PKG_INSTALL="sudo pacman -S --noconfirm"
elif command -v apt &> /dev/null; then
    PKG_MGR="apt"
    PKG_INSTALL="sudo apt install -y"
elif command -v dnf &> /dev/null; then
    PKG_MGR="dnf"
    PKG_INSTALL="sudo dnf install -y"
else
    echo "Warning: Unknown package manager. You may need to install dependencies manually."
    PKG_MGR="unknown"
fi

echo "Detected package manager: $PKG_MGR"

# Install system dependencies
echo ""
echo "=== Installing system dependencies ==="

case $PKG_MGR in
    pacman)
        $PKG_INSTALL python python-pip ffmpeg curl pup
        ;;
    apt)
        $PKG_INSTALL python3 python3-pip python3-venv ffmpeg curl
        # pup needs to be installed from go or downloaded
        if ! command -v pup &> /dev/null; then
            echo "Installing pup from GitHub releases..."
            curl -sL https://github.com/ericchiang/pup/releases/download/v0.4.0/pup_v0.4.0_linux_amd64.zip -o /tmp/pup.zip
            unzip -o /tmp/pup.zip -d /tmp
            sudo mv /tmp/pup /usr/local/bin/pup
            sudo chmod +x /usr/local/bin/pup
        fi
        ;;
    dnf)
        $PKG_INSTALL python3 python3-pip ffmpeg curl
        # Install pup
        if ! command -v pup &> /dev/null; then
            echo "Installing pup from GitHub releases..."
            curl -sL https://github.com/ericchiang/pup/releases/download/v0.4.0/pup_v0.4.0_linux_amd64.zip -o /tmp/pup.zip
            unzip -o /tmp/pup.zip -d /tmp
            sudo mv /tmp/pup /usr/local/bin/pup
            sudo chmod +x /usr/local/bin/pup
        fi
        ;;
    *)
        echo "Please install manually: python3, pip, ffmpeg, curl, pup"
        ;;
esac

# Install uv (Python package manager) if not present
if ! command -v uv &> /dev/null; then
    echo ""
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install uv globally for systemd to use
if [[ ! -f /usr/local/bin/uv ]]; then
    sudo cp "$HOME/.local/bin/uv" /usr/local/bin/uv
    sudo chmod +x /usr/local/bin/uv
fi

# Install pocket-tts as a uv tool (for CLI access)
echo ""
echo "=== Installing pocket-tts ==="
uv tool install pocket-tts || uv tool upgrade pocket-tts

# Create reeder user if it doesn't exist
echo ""
echo "=== Setting up reeder user ==="
if ! id -u reeder &> /dev/null; then
    sudo useradd --system --home-dir "$INSTALL_DIR" --shell /usr/sbin/nologin reeder
    echo "Created reeder user"
else
    echo "User reeder already exists"
fi

# Create directory structure
echo ""
echo "=== Creating directory structure ==="
sudo mkdir -p "$INSTALL_DIR"/{inbox,processing,done,www/audio,voices,var,bin}
sudo cp "$SCRIPT_DIR/bin/process-job" "$INSTALL_DIR/bin/"
sudo cp "$SCRIPT_DIR/bin/update-feed" "$INSTALL_DIR/bin/"
sudo cp "$SCRIPT_DIR/bin/submit-url" "$INSTALL_DIR/bin/"
sudo cp "$SCRIPT_DIR/bin/submit-text" "$INSTALL_DIR/bin/"
sudo cp "$SCRIPT_DIR/bin/reeder-status" "$INSTALL_DIR/bin/"
sudo chmod +x "$INSTALL_DIR/bin/"*

# Copy pyproject.toml for uv
sudo cp "$SCRIPT_DIR/pyproject.toml" "$INSTALL_DIR/pyproject.toml"

# Copy config if not exists
if [[ ! -f "$INSTALL_DIR/config.toml" ]]; then
    sudo cp "$SCRIPT_DIR/config.toml" "$INSTALL_DIR/config.toml"
    echo "Copied default config.toml"
else
    echo "config.toml already exists, skipping"
fi

# Set ownership
sudo chown -R reeder:reeder "$INSTALL_DIR"

# Install Python dependencies with uv
echo ""
echo "=== Installing Python dependencies ==="
cd "$INSTALL_DIR"
sudo -u reeder /usr/local/bin/uv sync
echo "Python dependencies installed to $INSTALL_DIR/.venv"

# Copy systemd units
echo ""
echo "=== Installing systemd units ==="
sudo cp "$SCRIPT_DIR/systemd/reeder.path" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/systemd/reeder.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/systemd/reeder.timer" /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start services
echo ""
echo "=== Enabling services ==="
sudo systemctl enable reeder.path
sudo systemctl start reeder.path
echo "reeder.path enabled and started"

# Optional: enable timer as fallback
# sudo systemctl enable reeder.timer
# sudo systemctl start reeder.timer

echo ""
echo "=== Installation complete ==="
echo ""
echo "Next steps:"
echo "1. Edit $INSTALL_DIR/config.toml to configure:"
echo "   - base_url (your Tailscale hostname)"
echo "   - default_voice (copy a voice file to $INSTALL_DIR/voices/)"
echo ""
echo "2. Add a voice file:"
echo "   sudo cp your-voice.wav $INSTALL_DIR/voices/default.wav"
echo "   sudo chown reeder:reeder $INSTALL_DIR/voices/default.wav"
echo ""
echo "3. Convert voice to safetensors (faster loading, recommended):"
echo "   # Note: Uses GitHub version of pocket-tts for safetensors support"
echo "   sudo -u reeder uvx --from git+https://github.com/kyutai-labs/pocket-tts.git pocket-tts export-voice $INSTALL_DIR/voices/default.wav $INSTALL_DIR/voices/default.safetensors --truncate"
echo "   # Then update config.toml: default_voice = \"default.safetensors\""
echo ""
echo "3. (Optional) Set up Caddy for HTTPS:"
echo "   sudo cp $SCRIPT_DIR/Caddyfile /etc/caddy/Caddyfile"
echo "   sudo systemctl enable --now caddy"
echo ""
echo "4. Test with a job:"
echo "   echo '{\"type\":\"text\",\"text\":\"Hello world\",\"title\":\"Test\"}' | sudo -u reeder tee $INSTALL_DIR/inbox/\$(date +%s)-test.json"
echo ""
echo "5. Monitor:"
echo "   journalctl -u reeder -f"
echo "   cat $INSTALL_DIR/var/status.txt"
