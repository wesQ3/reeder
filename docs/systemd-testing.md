# Testing Systemd Locally

You can test the systemd automation locally using **user services** before deploying to your server.

## Setup

Run the dev setup script:

```bash
./dev-setup.sh
```

This will:
1. Install Python dependencies with `uv sync`
2. Create necessary directories
3. Install systemd user service files to `~/.config/systemd/user/`
4. Enable and start `reeder.path` to watch for jobs

## Usage

### Submit a job

The path watcher will automatically process it:

```bash
export REEDER_CONFIG=$PWD/config.dev.toml
bin/submit-url https://example.com/article "Test Article"
```

### Monitor

```bash
# Watch logs in real-time
journalctl --user -u reeder -f

# Check service status
systemctl --user status reeder.path
systemctl --user status reeder.service

# Check queue status
bin/reeder-status
```

### Manual control

```bash
# Process jobs manually (bypasses path watcher)
systemctl --user start reeder.service

# Stop watching for jobs
systemctl --user stop reeder.path

# Restart watching
systemctl --user restart reeder.path

# Disable on login
systemctl --user disable reeder.path
```

## How it works

- **reeder.path** - Watches `inbox/` using `inotify`, triggers service when files appear
- **reeder.service** - Runs `bin/process-job` once per trigger (oneshot)
- User services run as your user (no `sudo` needed)
- Uses `%h` (home directory) expansion in unit files
- Logs to user journal: `journalctl --user -u reeder`

## Differences from system service

| Aspect | User Service | System Service |
|--------|-------------|----------------|
| Install location | `~/.config/systemd/user/` | `/etc/systemd/system/` |
| Command | `systemctl --user` | `sudo systemctl` |
| Runs as | Your user | `reeder` user |
| Auto-start | On login | On boot |
| Data location | `~/code/reeder/` | `/var/lib/reeder/` |

## Cleanup

```bash
# Stop and disable
systemctl --user stop reeder.path
systemctl --user disable reeder.path

# Remove service files
rm ~/.config/systemd/user/reeder.{path,service,timer}
systemctl --user daemon-reload
```
