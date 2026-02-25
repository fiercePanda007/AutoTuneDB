import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Dict, Any


def setup_logger(config: Dict[str, Any]) -> None:
    """
    Setup logging configuration for the entire system.
    
    Args:
        config: Configuration dictionary containing logging settings
    """
    log_config = config.get('logging', {})
    paths_config = config.get('paths', {})
    
    # Create logs directory
    log_dir = Path(paths_config.get('logs_dir', './data/logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get logging level
    level_str = log_config.get('level', 'INFO').upper()
    level = getattr(logging, level_str, logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt=log_config.get('format', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if log_config.get('console_output', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file = log_dir / 'optimizer.log'
    rotation_setting = log_config.get('rotation', '1 day')
    
    if 'day' in rotation_setting:
        days = int(rotation_setting.split()[0])
        file_handler = TimedRotatingFileHandler(
            log_file,
            when='D',
            interval=days,
            backupCount=int(log_config.get('retention', '30 days').split()[0])
        )
    else:
        max_bytes = 10 * 1024 * 1024  # 10 MB
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=10
        )
    
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_log_file = log_dir / 'errors.log'
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Log startup message
    root_logger.info("="*60)
    root_logger.info("Logging system initialized")
    root_logger.info(f"Log level: {level_str}")
    root_logger.info(f"Log directory: {log_dir}")
    root_logger.info("="*60)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name for the logger (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for temporary log level changes."""
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.old_level = None
        
    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)
        return False