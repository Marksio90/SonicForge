"""Real-time Anomaly Detection using Isolation Forest.

Detects anomalies in:
- System metrics (CPU, memory, latency)
- Application metrics (error rates, response times)
- Business metrics (track quality, generation success rate)
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, List
import structlog
import asyncio

logger = structlog.get_logger(__name__)


class AnomalyDetector:
    """Detect anomalies in system and application metrics."""
    
    def __init__(self, contamination: float = 0.05):
        """Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (default 5%)
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        self.metrics_history: List[Dict] = []
        self.is_trained = False
        self.min_samples = 100  # Minimum samples before training
        
    async def add_metrics(self, metrics: Dict) -> bool:
        """Add metrics sample and check for anomalies.
        
        Args:
            metrics: Dictionary of metric values
        
        Returns:
            True if anomaly detected, False otherwise
        """
        self.metrics_history.append(metrics)
        
        # Train model when we have enough samples
        if len(self.metrics_history) >= self.min_samples and not self.is_trained:
            await self._train_model()
        
        # Check for anomaly if model is trained
        if self.is_trained:
            is_anomaly = await self._check_anomaly(metrics)
            if is_anomaly:
                await self._handle_anomaly(metrics)
                return True
        
        # Keep only recent history (last 1000 samples)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return False
    
    async def _train_model(self):
        """Train the anomaly detection model."""
        logger.info("training_anomaly_detector", samples=len(self.metrics_history))
        
        X = self._prepare_features(self.metrics_history)
        
        # Run in thread pool to avoid blocking
        await asyncio.to_thread(self.model.fit, X)
        
        self.is_trained = True
        logger.info("anomaly_detector_trained")
    
    async def _check_anomaly(self, metrics: Dict) -> bool:
        """Check if current metrics are anomalous."""
        X = self._prepare_features([metrics])
        
        # Predict (-1 = anomaly, 1 = normal)
        prediction = await asyncio.to_thread(self.model.predict, X)
        
        return prediction[0] == -1
    
    def _prepare_features(self, metrics_list: List[Dict]) -> np.ndarray:
        """Convert metrics to feature matrix."""
        features = []
        for m in metrics_list:
            feature_vector = [
                m.get("cpu_percent", 0),
                m.get("memory_percent", 0),
                m.get("api_latency_ms", 0),
                m.get("error_rate", 0),
                m.get("queue_length", 0),
                m.get("db_query_time_ms", 0),
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    async def _handle_anomaly(self, metrics: Dict):
        """Handle detected anomaly."""
        logger.warning(
            "anomaly_detected",
            cpu=metrics.get("cpu_percent"),
            memory=metrics.get("memory_percent"),
            latency=metrics.get("api_latency_ms"),
            error_rate=metrics.get("error_rate"),
        )
        
        # Send alert
        await self._send_alert(metrics)
    
    async def _send_alert(self, metrics: Dict):
        """Send alert notification."""
        # TODO: Integrate with alerting system (Slack, PagerDuty, etc.)
        pass


class MetricsCollector:
    """Collect system and application metrics for anomaly detection."""
    
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
    
    async def collect_and_check(self):
        """Collect current metrics and check for anomalies."""
        import psutil
        
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "api_latency_ms": 0,  # TODO: Get from metrics
            "error_rate": 0,  # TODO: Get from metrics
            "queue_length": 0,  # TODO: Get from Redis
            "db_query_time_ms": 0,  # TODO: Get from metrics
        }
        
        is_anomaly = await self.anomaly_detector.add_metrics(metrics)
        
        return {
            "metrics": metrics,
            "is_anomaly": is_anomaly,
        }


# Global instance
metrics_collector = MetricsCollector()
