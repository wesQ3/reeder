#!/bin/bash
# Setup Reeder for local development with systemd user services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Reeder Development Setup ==="
echo ""

# Ensure directories exist
echo "Creating directories..."
mkdir -p "$SCRIPT_DIR"/{inbox,processing,done,www/audio,voices,var}

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."
cd "$SCRIPT_DIR"
uv sync

echo ""
echo "=== Installing systemd user services ==="

# Create user systemd directory
mkdir -p ~/.config/systemd/user

# Update the service files with actual home directory
# (systemd %h works at runtime, but we need absolute paths for some things)
HOME_ESC=$(echo "$HOME" | sed 's/\//\\\//g')
REEDER_DIR="$SCRIPT_DIR"
REEDER_ESC=$(echo "$REEDER_DIR" | sed 's/\//\\\//g')

# Copy and customize for current user's paths
for file in "$SCRIPT_DIR/systemd/user"/*.{path,service,timer}; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")
        cp "$file" ~/.config/systemd/user/
        echo "Installed: $filename"
    fi
done

# Reload systemd user daemon
systemctl --user daemon-reload

# Enable and start the path watcher
echo ""
echo "=== Enabling reeder.path ==="
systemctl --user enable reeder.path
systemctl --user start reeder.path

echo ""
echo "=== Setup complete ==="
echo ""
echo "Usage:"
echo ""
echo "  # Submit jobs"
echo "  export REEDER_CONFIG=$SCRIPT_DIR/config.dev.toml"
echo "  $SCRIPT_DIR/bin/submit-url https://example.com/article"
echo ""
echo "  # Check status"
echo "  systemctl --user status reeder.path"
echo "  systemctl --user status reeder.service"
echo ""
echo "  # View logs"
echo "  journalctl --user -u reeder -f"
echo ""
echo "  # Manual trigger"
echo "  systemctl --user start reeder.service"
echo ""
echo "  # Stop watching"
echo "  systemctl --user stop reeder.path"
echo ""
echo "  # Check status"
echo "  $SCRIPT_DIR/bin/reeder-status"
echo ""
