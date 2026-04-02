"""Logging utilities and token tracking."""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class LogConfig:
    """Configuration for logging."""
    log_dir: str = "logs"
    level: str = "INFO"
    log_api_calls: bool = True
    log_tree_states: bool = True
    rotation: str = "10 MB"
    retention: str = "7 days"


def setup_logger(config: Optional[LogConfig] = None) -> None:
    """Setup loguru logger with file and console handlers."""
    config = config or LogConfig()
    
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.remove()
    
    logger.add(
        sys.stderr,
        level=config.level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation=config.rotation,
        retention=config.retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    
    if config.log_api_calls:
        logger.add(
            log_dir / "api_calls_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            rotation=config.rotation,
            retention=config.retention,
            filter=lambda record: "api" in record["extra"],
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        )
    
    logger.info("Logger initialized")


@dataclass
class TokenUsage:
    """Token usage record."""
    timestamp: float
    input_tokens: int
    output_tokens: int
    model: str
    request_type: str
    latency_ms: float
    cached: bool = False


class TokenTracker:
    """Track token usage across API calls."""
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self._usage_history: List[TokenUsage] = []
        self._total_input = 0
        self._total_output = 0
        self._total_cached = 0
    
    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        request_type: str,
        latency_ms: float,
        cached: bool = False,
    ) -> None:
        """Record token usage."""
        usage = TokenUsage(
            timestamp=time.time(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            request_type=request_type,
            latency_ms=latency_ms,
            cached=cached,
        )
        
        self._usage_history.append(usage)
        self._total_input += input_tokens
        self._total_output += output_tokens
        
        if cached:
            self._total_cached += input_tokens + output_tokens
        
        logger.bind(api=True).debug(
            f"API call: {request_type} | "
            f"input={input_tokens} | output={output_tokens} | "
            f"latency={latency_ms:.0f}ms | cached={cached}"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        if not self._usage_history:
            return {"total_input": 0, "total_output": 0, "total": 0}
        
        avg_latency = sum(u.latency_ms for u in self._usage_history) / len(self._usage_history)
        
        return {
            "total_calls": len(self._usage_history),
            "total_input_tokens": self._total_input,
            "total_output_tokens": self._total_output,
            "total_tokens": self._total_input + self._total_output,
            "cached_tokens": self._total_cached,
            "avg_latency_ms": avg_latency,
            "by_request_type": self._group_by_type(),
        }
    
    def _group_by_type(self) -> Dict[str, Dict[str, int]]:
        """Group usage by request type."""
        groups: Dict[str, Dict[str, int]] = {}
        
        for usage in self._usage_history:
            if usage.request_type not in groups:
                groups[usage.request_type] = {
                    "count": 0,
                    "input": 0,
                    "output": 0,
                }
            
            groups[usage.request_type]["count"] += 1
            groups[usage.request_type]["input"] += usage.input_tokens
            groups[usage.request_type]["output"] += usage.output_tokens
        
        return groups
    
    def get_hourly_usage(self) -> Dict[str, int]:
        """Get hourly token distribution."""
        hourly: Dict[str, int] = {}
        
        for usage in self._usage_history:
            hour = datetime.fromtimestamp(usage.timestamp).strftime("%Y-%m-%d %H:00")
            hourly[hour] = hourly.get(hour, 0) + usage.input_tokens + usage.output_tokens
        
        return hourly
    
    def save(self, filename: str = "token_usage.json") -> None:
        """Save usage history to file."""
        if not self.save_dir:
            return
        
        path = self.save_dir / filename
        data = {
            "summary": self.get_summary(),
            "history": [
                {
                    "timestamp": u.timestamp,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "model": u.model,
                    "request_type": u.request_type,
                    "latency_ms": u.latency_ms,
                    "cached": u.cached,
                }
                for u in self._usage_history
            ],
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Token usage saved to {path}")
    
    def reset(self) -> None:
        """Reset tracker."""
        self._usage_history.clear()
        self._total_input = 0
        self._total_output = 0
        self._total_cached = 0


class ExperimentLogger:
    """Logger for experiment tracking."""
    
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = time.time()
        self._log_file = self.log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self._entries: List[Dict] = []
        
        logger.info(f"Experiment logger created: {self._log_file}")
    
    def log(self, event: str, data: Optional[Dict] = None) -> None:
        """Log an event."""
        entry = {
            "timestamp": time.time(),
            "elapsed": time.time() - self.start_time,
            "event": event,
            "data": data or {},
        }
        
        self._entries.append(entry)
        
        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        logger.debug(f"Logged event: {event}")
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics."""
        self.log("metrics", metrics)
    
    def log_checkpoint(self, name: str, data: Dict) -> None:
        """Log checkpoint."""
        self.log(f"checkpoint_{name}", data)
    
    def finalize(self) -> Dict:
        """Finalize and return summary."""
        elapsed = time.time() - self.start_time
        
        summary = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time,
            "elapsed_seconds": elapsed,
            "total_entries": len(self._entries),
        }
        
        self.log("experiment_complete", summary)
        
        return summary
