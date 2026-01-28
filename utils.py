import os
import sys
import logging
from datetime import datetime
from typing import Optional

def build_logger(log_dir: str,
                 log_name: Optional[str] = None,
                 level: int = logging.INFO,
                 console: bool = True) -> logging.Logger:

    os.makedirs(log_dir, exist_ok=True)

    if log_name is None:
        log_name = f"{datetime.now():%Y%m%d_%H%M%S}.log"
    log_path = os.path.join(log_dir, log_name)

    handlers = [logging.FileHandler(log_path, mode='w', encoding='utf-8')]
    if console:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers
    )
    return logging.getLogger(__name__)
