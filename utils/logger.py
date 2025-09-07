"""
Logging configuration for the ML service
"""

import logging
import sys
from datetime import datetime
from typing import Optional
from loguru import logger as loguru_logger

def setup_logger(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logger with custom configuration
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string
        log_file: Optional log file path
    
    Returns:
        Configured logger instance
    """
    
    # Remove default handler
    loguru_logger.remove()
    
    # Default format
    if not format_string:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Console handler
    loguru_logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File handler if specified
    if log_file:
        loguru_logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    # Create standard logging adapter for compatibility
    class LoguruHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = loguru_logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            loguru_logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Set up standard logging to use loguru
    logging.basicConfig(handlers=[LoguruHandler()], level=0, force=True)
    
    return logging.getLogger(__name__)

def log_ml_operation(
    operation: str,
    duration_ms: float,
    success: bool,
    details: Optional[dict] = None
):
    """
    Log ML operation with structured data
    
    Args:
        operation: Name of the ML operation
        duration_ms: Operation duration in milliseconds
        success: Whether operation was successful
        details: Additional operation details
    """
    log_data = {
        "operation": operation,
        "duration_ms": round(duration_ms, 2),
        "success": success,
        "timestamp": datetime.now().isoformat()
    }
    
    if details:
        log_data.update(details)
    
    if success:
        loguru_logger.info(f"ML Operation: {operation}", **log_data)
    else:
        loguru_logger.error(f"ML Operation Failed: {operation}", **log_data)

def log_model_performance(
    model_name: str,
    metrics: dict,
    dataset_info: Optional[dict] = None
):
    """
    Log model performance metrics
    
    Args:
        model_name: Name of the model
        metrics: Performance metrics dictionary
        dataset_info: Information about the dataset used
    """
    log_data = {
        "model": model_name,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    if dataset_info:
        log_data["dataset"] = dataset_info
    
    loguru_logger.info(f"Model Performance: {model_name}", **log_data)

def log_api_request(
    endpoint: str,
    method: str,
    status_code: int,
    duration_ms: float,
    user_id: Optional[str] = None,
    request_size: Optional[int] = None
):
    """
    Log API request details
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        status_code: Response status code
        duration_ms: Request duration in milliseconds
        user_id: Optional user identifier
        request_size: Optional request size in bytes
    """
    log_data = {
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        "timestamp": datetime.now().isoformat()
    }
    
    if user_id:
        log_data["user_id"] = user_id
    
    if request_size:
        log_data["request_size"] = request_size
    
    level = "info" if 200 <= status_code < 400 else "warning" if status_code < 500 else "error"
    
    loguru_logger.log(level.upper(), f"API Request: {method} {endpoint}", **log_data)

def log_cache_operation(
    operation: str,
    key: str,
    hit: Optional[bool] = None,
    size: Optional[int] = None
):
    """
    Log cache operation
    
    Args:
        operation: Cache operation (get, set, delete, etc.)
        key: Cache key
        hit: Whether it was a cache hit (for get operations)
        size: Size of cached data
    """
    log_data = {
        "operation": operation,
        "key": key,
        "timestamp": datetime.now().isoformat()
    }
    
    if hit is not None:
        log_data["hit"] = hit
    
    if size is not None:
        log_data["size"] = size
    
    loguru_logger.debug(f"Cache {operation}: {key}", **log_data)

class MLLogger:
    """
    Specialized logger for ML operations with context
    """
    
    def __init__(self, context: str):
        self.context = context
        self.logger = loguru_logger.bind(context=context)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(message, **kwargs)
    
    def log_prediction(
        self,
        model_name: str,
        input_text: str,
        prediction: str,
        confidence: float,
        processing_time_ms: float
    ):
        """Log model prediction"""
        self.logger.info(
            f"Prediction: {model_name}",
            input_length=len(input_text),
            prediction=prediction,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now().isoformat()
        )
    
    def log_training_start(self, model_name: str, dataset_size: int):
        """Log training start"""
        self.logger.info(
            f"Training Started: {model_name}",
            dataset_size=dataset_size,
            timestamp=datetime.now().isoformat()
        )
    
    def log_training_complete(
        self,
        model_name: str,
        training_time_seconds: float,
        final_metrics: dict
    ):
        """Log training completion"""
        self.logger.info(
            f"Training Complete: {model_name}",
            training_time_seconds=training_time_seconds,
            metrics=final_metrics,
            timestamp=datetime.now().isoformat()
        )
    
    def log_error_with_context(
        self,
        error: Exception,
        operation: str,
        context_data: Optional[dict] = None
    ):
        """Log error with additional context"""
        error_data = {
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
        if context_data:
            error_data.update(context_data)
        
        self.logger.error(f"Error in {operation}: {error}", **error_data)

# Global logger instance
ml_logger = MLLogger("ML_SERVICE")