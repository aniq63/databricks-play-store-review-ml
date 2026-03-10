import sys
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# ── Constants ──────────────────────────────────────────────────────────────────
LOG_DIR      = 'logs'
LOG_FILE     = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024   # 5 MB
BACKUP_COUNT = 3

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_dir_path  = os.path.join(PROJECT_ROOT, LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)

# ── Silence noisy third-party loggers ─────────────────────────────────────────
# These flood the log with hundreds of DEBUG lines per operation.
_QUIET_LOGGERS = [
    "urllib3",
    "urllib3.connectionpool",
    "databricks.sdk",
    "databricks.sdk.mixins.files",
    "git",
    "git.cmd",
    "git.util",
    "mlflow",
    "py4j",
    "httpx",
]
for _name in _QUIET_LOGGERS:
    logging.getLogger(_name).setLevel(logging.WARNING)


def configure_logger(max_log_size: int = MAX_LOG_SIZE, backup_count: int = BACKUP_COUNT):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)     # root at INFO — DEBUG never reaches handlers

    # Clear existing handlers to avoid duplicates on re-import
    logger.handlers.clear()

    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    # ── File handler — INFO only, rotating, UTF-8 ─────────────────────────────
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=max_log_size,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # ── Console handler — INFO only, UTF-8 to avoid Windows cp1252 errors ─────
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Configure on import
logging = configure_logger()
