"""
Comprehensive error handling and logging configuration
"""
import logging
import sys
import traceback
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional
import functools


class TradingBotError(Exception):
    """Base exception for trading bot errors"""
    pass


class ConfigurationError(TradingBotError):
    """Configuration related errors"""
    pass


class ExchangeError(TradingBotError):
    """Exchange connectivity/API errors"""
    pass


class StrategyError(TradingBotError):
    """Trading strategy errors"""
    pass


class DatabaseError(TradingBotError):
    """Database operation errors"""
    pass


class ValidationError(TradingBotError):
    """Data validation errors"""
    pass


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    max_bytes: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('trading_bot')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Error file handler (separate file for errors)
    if log_file:
        error_file = log_file.parent / f"{log_file.stem}_errors.log"
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
    
    return logger


def handle_exception(logger: logging.Logger):
    """
    Decorator for handling exceptions with logging
    
    Usage:
        @handle_exception(logger)
        def my_function():
            # code that might raise exceptions
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except TradingBotError as e:
                logger.error(f"{func.__name__} failed: {str(e)}")
                raise
            except Exception as e:
                logger.exception(f"Unexpected error in {func.__name__}: {str(e)}")
                raise TradingBotError(f"Unexpected error in {func.__name__}") from e
        return wrapper
    return decorator


def handle_exception_async(logger: logging.Logger):
    """
    Decorator for handling exceptions in async functions
    
    Usage:
        @handle_exception_async(logger)
        async def my_async_function():
            # async code that might raise exceptions
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except TradingBotError as e:
                logger.error(f"{func.__name__} failed: {str(e)}")
                raise
            except Exception as e:
                logger.exception(f"Unexpected error in {func.__name__}: {str(e)}")
                raise TradingBotError(f"Unexpected error in {func.__name__}") from e
        return wrapper
    return decorator


class ErrorLogger:
    """Centralized error logging and reporting"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_count = {}
        self.last_errors = []
        self.max_recent_errors = 100
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context and tracking"""
        error_type = type(error).__name__
        
        # Count errors by type
        self.error_count[error_type] = self.error_count.get(error_type, 0) + 1
        
        # Store recent errors
        error_info = {
            'type': error_type,
            'message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc()
        }
        self.last_errors.append(error_info)
        
        # Keep only recent errors
        if len(self.last_errors) > self.max_recent_errors:
            self.last_errors.pop(0)
        
        # Log the error
        self.logger.error(
            f"{error_type} in {context}: {str(error)}",
            extra={'error_info': error_info}
        )
    
    def get_error_statistics(self):
        """Get error statistics"""
        return {
            'total_errors': sum(self.error_count.values()),
            'by_type': self.error_count.copy(),
            'recent_count': len(self.last_errors)
        }
    
    def get_recent_errors(self, count: int = 10):
        """Get recent errors"""
        return self.last_errors[-count:]


class CircuitBreaker:
    """Circuit breaker pattern for external services"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        logger: Optional[logging.Logger] = None
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open
        self.logger = logger or logging.getLogger(__name__)
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half_open'
                self.logger.info("Circuit breaker entering half-open state")
            else:
                raise TradingBotError("Circuit breaker is open - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func, *args, **kwargs):
        """Execute async function with circuit breaker protection"""
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half_open'
                self.logger.info("Circuit breaker entering half-open state")
            else:
                raise TradingBotError("Circuit breaker is open - service unavailable")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self):
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == 'half_open':
            self.logger.info("Circuit breaker closed - service recovered")
            self.state = 'closed'
        
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            if self.state != 'open':
                self.logger.error(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )
                self.state = 'open'


# Global error logger instance
_error_logger: Optional[ErrorLogger] = None


def get_error_logger() -> ErrorLogger:
    """Get global error logger instance"""
    global _error_logger
    if _error_logger is None:
        logger = logging.getLogger('trading_bot')
        _error_logger = ErrorLogger(logger)
    return _error_logger


# Global circuit breakers for external services
_circuit_breakers = {}


def get_circuit_breaker(service_name: str) -> CircuitBreaker:
    """Get circuit breaker for a service"""
    if service_name not in _circuit_breakers:
        logger = logging.getLogger(f'circuit_breaker.{service_name}')
        _circuit_breakers[service_name] = CircuitBreaker(logger=logger)
    return _circuit_breakers[service_name]
