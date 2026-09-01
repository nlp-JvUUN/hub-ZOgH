"""Logging utilities for GRPO training."""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from collections import defaultdict
from pathlib import Path


class TrainingLogger:
    """Logger for training metrics and events.
    
    Supports:
    - TensorBoard-like logging
    - JSON metrics files
    - Console output
    """
    
    def __init__(self,
                 log_dir: str = "./logs",
                 experiment_name: Optional[str] = None):
        """Initialize the logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = self.log_dir / self.experiment_name
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.scalar_metrics = {}
        
        # File paths
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.config_file = self.log_dir / "config.json"
        self.events_file = self.log_dir / "events.jsonl"
        
        # Initialize files
        if not self.metrics_file.exists():
            self.metrics_file.touch()
        if not self.events_file.exists():
            self.events_file.touch()
        
        # Create logger
        self.logger = logging.getLogger(f"grpo_trainer.{self.experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logger initialized at: {self.log_dir}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for a training step.
        
        Args:
            metrics: Dictionary of metric name to value
            step: Training step number
        """
        # Store scalar metrics
        for name, value in metrics.items():
            self.scalar_metrics[f"{name}/step_{step}"] = value
            self.metrics[name].append((step, value))
        
        # Write to JSONL file
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event (e.g., evaluation, checkpoint).
        
        Args:
            event_type: Type of event
            data: Event data
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        
        with open(self.events_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.info(f"Event [{event_type}]: {data}")
    
    def log_checkpoint(self, checkpoint_info: Dict[str, Any]):
        """Log a checkpoint event.
        
        Args:
            checkpoint_info: Checkpoint metadata
        """
        self.log_event("checkpoint", checkpoint_info)
    
    def log_evaluation(self, eval_results: Dict[str, float]):
        """Log evaluation results.
        
        Args:
            eval_results: Evaluation metrics
        """
        self.log_event("evaluation", eval_results)
        
        # Also log as metrics
        for name, value in eval_results.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"Eval [{name}]: {value}")
    
    def get_metric_history(self, metric_name: str) -> list:
        """Get history of a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of (step, value) tuples
        """
        return self.metrics.get(metric_name, [])
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the latest values for all metrics.
        
        Returns:
            Dictionary of latest metric values
        """
        latest = {}
        for name, history in self.metrics.items():
            if history:
                latest[name] = history[-1][1]
        return latest
    
    def save_metrics_summary(self):
        """Save a summary of all metrics to file."""
        summary = {
            "metrics": dict(self.metrics),
            "latest": self.get_latest_metrics(),
            "timestamp": datetime.now().isoformat()
        }
        
        summary_file = self.log_dir / "metrics_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Metrics summary saved to: {summary_file}")
    
    def close(self):
        """Close the logger and save summary."""
        self.save_metrics_summary()
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


def setup_logging(level: str = "INFO"):
    """Setup basic logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


class MetricTracker:
    """Track and compute running statistics of metrics."""
    
    def __init__(self):
        self.values = []
        self.running_sum = 0.0
        self.running_sq_sum = 0.0
        self.count = 0
    
    def add(self, value: float):
        """Add a value to the tracker."""
        self.values.append(value)
        self.running_sum += value
        self.running_sq_sum += value * value
        self.count += 1
    
    def mean(self) -> float:
        """Get the mean of all values."""
        if self.count == 0:
            return 0.0
        return self.running_sum / self.count
    
    def std(self) -> float:
        """Get the standard deviation of all values."""
        if self.count < 2:
            return 0.0
        variance = (self.running_sq_sum - self.running_sum**2 / self.count) / (self.count - 1)
        return variance ** 0.5
    
    def last(self, n: int = 1) -> float:
        """Get the last n values."""
        if not self.values:
            return 0.0
        return sum(self.values[-n:]) / min(n, len(self.values))
    
    def reset(self):
        """Reset all statistics."""
        self.values = []
        self.running_sum = 0.0
        self.running_sq_sum = 0.0
        self.count = 0
    
    def summary(self) -> Dict[str, float]:
        """Get a summary of statistics."""
        if self.count == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        
        return {
            "mean": self.mean(),
            "std": self.std(),
            "min": min(self.values),
            "max": max(self.values),
            "count": self.count
        }
