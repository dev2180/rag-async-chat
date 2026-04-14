import os
import logging
from logging.handlers import RotatingFileHandler
from app.config import setup_logging, LOG_DIR, LOG_FILE

def test_setup_logging_creates_file_and_handlers(tmp_path, monkeypatch):
    # Reroute LOG_DIR to a temporary directory so we don't spam the actual logs folder during testing
    test_log_dir = str(tmp_path / "logs")
    monkeypatch.setattr("app.config.LOG_DIR", test_log_dir)
    
    setup_logging(console_level="WARNING")
    
    # Assert directory and file are created when log is emitted
    root_logger = logging.getLogger()
    
    # Needs to log first to create file usually, but we called exit_ok=True inside
    root_logger.warning("Test message")
    
    log_path = os.path.join(test_log_dir, LOG_FILE)
    assert os.path.exists(test_log_dir)
    assert os.path.exists(log_path)
    
    # Find the handlers
    handlers = root_logger.handlers
    assert len(handlers) == 2
    
    file_handler = next((h for h in handlers if isinstance(h, RotatingFileHandler)), None)
    console_handler = next((h for h in handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)), None)
    
    assert file_handler is not None
    assert console_handler is not None
    
    # Verify exact limits (5MB)
    assert file_handler.maxBytes == 5 * 1024 * 1024
    assert file_handler.backupCount == 3
    
    # Verify console level was respected
    assert console_handler.level == logging.WARNING
    
    # Cleanup handler so it doesn't lock files in windows
    for h in handlers:
        h.close()
        root_logger.removeHandler(h)
