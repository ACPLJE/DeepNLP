import logging
import sys
from pathlib import Path
from datetime import datetime

class Logger:
    """Custom logger class for the project."""
    
    def __init__(self, name, log_file=None, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if log_file is specified)
        if log_file is None:
            # Create default log file in logs directory
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"{name}_{timestamp}.log"
            
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, msg, *args, **kwargs):
        """Log info level message."""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """Log warning level message."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """Log error level message."""
        self.logger.error(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        """Log debug level message."""
        self.logger.debug(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        """Log critical level message."""
        self.logger.critical(msg, *args, **kwargs)

def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up and return a logger instance.
    
    Args:
        name (str): Name of the logger
        log_file (str, optional): Path to log file. If None, creates a timestamped file
        level (int, optional): Logging level. Defaults to logging.INFO
        
    Returns:
        Logger: Configured logger instance
    """
    return Logger(name, log_file, level)