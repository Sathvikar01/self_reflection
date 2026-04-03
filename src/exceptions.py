"""Custom exception hierarchy for self-reflection system."""

from typing import Optional, Any


class ReflectionError(Exception):
    """Base exception for all self-reflection errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class APIError(ReflectionError):
    """Errors related to API interactions."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        api_response: Optional[dict] = None
    ):
        details = {"status_code": status_code, "api_response": api_response}
        super().__init__(message, details)
        self.status_code = status_code
        self.api_response = api_response


class GenerationError(APIError):
    """Errors during text generation."""
    
    def __init__(self, message: str, model: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.model = model
        self.details["model"] = model


class EvaluationError(ReflectionError):
    """Errors during evaluation/scoring."""
    
    def __init__(self, message: str, step: Optional[str] = None):
        details = {"step": step} if step else {}
        super().__init__(message, details)
        self.step = step


class AnswerExtractionError(ReflectionError):
    """Errors during answer extraction."""
    pass


class DataValidationError(ReflectionError):
    """Errors in data validation."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        details = {"field": field, "value": value}
        super().__init__(message, details)
        self.field = field
        self.value = value


class ConfigurationError(ReflectionError):
    """Errors in configuration."""
    pass


class PipelineError(ReflectionError):
    """Errors in pipeline execution."""
    
    def __init__(self, message: str, stage: Optional[str] = None, **kwargs):
        details = {"stage": stage, **kwargs}
        super().__init__(message, details)
        self.stage = stage
