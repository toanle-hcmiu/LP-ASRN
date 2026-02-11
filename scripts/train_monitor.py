"""
Training Monitor — wraps training with crash/completion notifications.

Sends notifications via Discord webhook (free, no setup needed).
Also monitors GPU memory and auto-saves emergency checkpoint on OOM.

Usage:
    # Set your Discord webhook URL (create one in any Discord server → channel settings → integrations → webhooks)
    export DISCORD_WEBHOOK="https://discord.com/api/webhooks/YOUR_WEBHOOK_URL"

    # Run training with monitoring
    python scripts/train_monitor.py --stage 2 --resume outputs/run_XXX/best.pth --save-dir outputs/run_XXX

    # All arguments are passed through to train_progressive.py
    # You can also use a .env file or set the webhook in the script below
"""

import subprocess
import sys
import os
import time
import json
import platform
from pathlib import Path
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — Set your notification method here
# ═══════════════════════════════════════════════════════════════════════

# Option 1: Discord webhook (recommended — free, instant, mobile notifications)
# Create one: Discord server → channel settings → Integrations → Webhooks → New Webhook
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")

# Option 2: ntfy.sh (EASIEST — no signup, free, instant push notifications)
# Just pick any unique topic name and subscribe on your phone
# Install app: https://ntfy.sh (Android/iOS) → Subscribe to your topic
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")  # e.g. "lp-asrn-training-xyz123"

# Option 3: Telegram bot
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Option 4: Email via Gmail (requires app password, not regular password)
EMAIL_ADDRESS = os.environ.get("MONITOR_EMAIL", "")  # your.email@gmail.com
EMAIL_APP_PASSWORD = os.environ.get("MONITOR_EMAIL_PASSWORD", "")  # Gmail app password

# GPU memory warning threshold (%)
GPU_MEM_WARN_THRESHOLD = 90


def send_discord(message: str, color: int = 0x00FF00):
    """Send a Discord webhook notification."""
    if not DISCORD_WEBHOOK:
        return False
    try:
        import urllib.request
        payload = json.dumps({
            "embeds": [{
                "title": "🔧 LP-ASRN Training",
                "description": message,
                "color": color,
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": platform.node()},
            }]
        }).encode("utf-8")
        req = urllib.request.Request(
            DISCORD_WEBHOOK,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"[Monitor] Discord notification failed: {e}")
        return False


def send_ntfy(message: str, is_error: bool = False):
    """Send notification via ntfy.sh (free, no signup, instant push)."""
    if not NTFY_TOPIC:
        return False
    try:
        import urllib.request
        title = "❌ LP-ASRN CRASHED" if is_error else "✅ LP-ASRN Training"
        priority = "urgent" if is_error else "default"
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": priority,
                "Tags": "rotating_light" if is_error else "white_check_mark",
            },
        )
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"[Monitor] ntfy.sh notification failed: {e}")
        return False


def send_telegram(message: str):
    """Send a Telegram notification."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        import urllib.request
        import urllib.parse
        url = (f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
               f"/sendMessage?chat_id={TELEGRAM_CHAT_ID}"
               f"&text={urllib.parse.quote(message)}"
               f"&parse_mode=Markdown")
        urllib.request.urlopen(url, timeout=10)
        return True
    except Exception as e:
        print(f"[Monitor] Telegram notification failed: {e}")
        return False


def send_email(message: str, is_error: bool = False):
    """Send email notification via Gmail SMTP."""
    if not EMAIL_ADDRESS or not EMAIL_APP_PASSWORD:
        return False
    try:
        import smtplib
        from email.mime.text import MIMEText
        subject = "❌ LP-ASRN CRASHED" if is_error else "✅ LP-ASRN Training Update"
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = EMAIL_ADDRESS
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"[Monitor] Email notification failed: {e}")
        return False


def notify(message: str, is_error: bool = False):
    """Send notification via all configured channels."""
    prefix = "❌" if is_error else "✅"
    full_msg = f"{prefix} {message}"
    print(f"[Monitor] {full_msg}")

    # Clean message for plain-text channels (remove markdown)
    plain_msg = message.replace("**", "").replace("```", "").replace("`", "")

    sent = False
    sent = send_ntfy(plain_msg, is_error=is_error) or sent
    sent = send_discord(full_msg, color=0xFF0000 if is_error else 0x00FF00) or sent
    sent = send_telegram(full_msg) or sent
    sent = send_email(plain_msg, is_error=is_error) or sent

    if not sent:
        print("[Monitor] No notification channel configured.")
        print("[Monitor] Easiest: export NTFY_TOPIC='my-training-xyz' (then install ntfy app on phone)")


def get_gpu_info():
    """Get GPU memory usage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 3:
                    used, total, util = int(parts[0]), int(parts[1]), int(parts[2])
                    gpus.append({"used_mb": used, "total_mb": total, "util_pct": util,
                                 "mem_pct": round(100 * used / total, 1)})
            return gpus
    except Exception:
        pass
    return []


def find_latest_checkpoint(save_dir: str):
    """Find most recent checkpoint."""
    save_path = Path(save_dir)
    if not save_path.exists():
        return None
    for name in ["latest.pth", "best.pth", "emergency_latest.pth"]:
        p = save_path / name
        if p.exists():
            return str(p)
    return None


def main():
    args = sys.argv[1:]

    if not args or "--help" in args or "-h" in args:
        print(__doc__)
        print("All arguments are passed through to train_progressive.py")
        return

    # Parse save-dir from args for checkpoint monitoring
    save_dir = None
    for i, arg in enumerate(args):
        if arg == "--save-dir" and i + 1 < len(args):
            save_dir = args[i + 1]

    # Build the training command
    cmd = [sys.executable, "scripts/train_progressive.py"] + args

    print(f"[Monitor] Starting training: {' '.join(cmd)}")
    channels = []
    if NTFY_TOPIC: channels.append(f"ntfy({NTFY_TOPIC})")
    if DISCORD_WEBHOOK: channels.append("Discord")
    if TELEGRAM_BOT_TOKEN: channels.append("Telegram")
    if EMAIL_ADDRESS: channels.append(f"Email({EMAIL_ADDRESS})")
    print(f"[Monitor] Notifications: {', '.join(channels) if channels else 'NONE — set NTFY_TOPIC!'}")

    start_time = time.time()
    notify(f"**Training started**\n"
           f"```\n{' '.join(args)}\n```\n"
           f"Host: `{platform.node()}`")

    # Run training as subprocess
    try:
        process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        # Wait for process, checking GPU periodically
        while process.poll() is None:
            time.sleep(30)  # Check every 30 seconds

            # Monitor GPU memory
            gpus = get_gpu_info()
            for i, gpu in enumerate(gpus):
                if gpu["mem_pct"] > GPU_MEM_WARN_THRESHOLD:
                    notify(f"⚠️ **GPU {i} memory high**: {gpu['used_mb']}MB / {gpu['total_mb']}MB "
                           f"({gpu['mem_pct']}%)", is_error=True)

        exit_code = process.returncode
        elapsed = timedelta(seconds=int(time.time() - start_time))

        if exit_code == 0:
            # Success
            gpu_info = get_gpu_info()
            gpu_str = ", ".join(f"GPU{i}: {g['mem_pct']}%" for i, g in enumerate(gpu_info)) if gpu_info else "N/A"

            notify(f"**Training completed successfully** ✅\n"
                   f"Duration: `{elapsed}`\n"
                   f"GPU memory: {gpu_str}")
        else:
            # Crash
            checkpoint_msg = ""
            if save_dir:
                ckpt = find_latest_checkpoint(save_dir)
                checkpoint_msg = f"\nLatest checkpoint: `{ckpt}`" if ckpt else "\nNo checkpoint found!"

            notify(f"**Training CRASHED** (exit code {exit_code})\n"
                   f"Duration: `{elapsed}`\n"
                   f"Signal: {'SIGKILL (OOM?)' if exit_code == -9 else 'SIGTERM' if exit_code == -15 else f'code {exit_code}'}"
                   f"{checkpoint_msg}\n"
                   f"Resume: `python scripts/train_monitor.py {' '.join(args)}`",
                   is_error=True)

    except KeyboardInterrupt:
        elapsed = timedelta(seconds=int(time.time() - start_time))
        process.terminate()
        notify(f"**Training stopped by user** (Ctrl+C)\nDuration: `{elapsed}`")
        process.wait()

    except Exception as e:
        elapsed = timedelta(seconds=int(time.time() - start_time))
        notify(f"**Monitor error**: `{e}`\nDuration: `{elapsed}`", is_error=True)
        raise


if __name__ == "__main__":
    main()

