"""
Medical AI Monitoring and Analytics System

Comprehensive monitoring for:
- Performance metrics and response times
- Medical accuracy and safety metrics  
- User interaction patterns
- System health and component status
- Error tracking and alerting
"""

import logging
import time
import json
import threading
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import statistics
import os

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    timestamp: float
    component: str
    operation: str
    duration: float
    success: bool
    metadata: Dict[str, Any]

@dataclass
class MedicalAccuracyMetric:
    """Medical accuracy measurement"""
    timestamp: float
    query_type: str
    entities_found: int
    predictions_count: int
    confidence: float
    uncertainty: float
    safety_flags: List[str]
    user_feedback: Optional[str] = None

@dataclass
class SystemHealthMetric:
    """System health status"""
    timestamp: float
    component: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    cpu_usage: float
    memory_usage: float
    error_rate: float
    response_time: float

class MetricsCollector:
    """
    Collects and aggregates metrics from the medical AI pipeline
    """
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector
        
        Args:
            retention_hours: How long to keep metrics in memory
        """
        self.retention_hours = retention_hours
        self.retention_seconds = retention_hours * 3600
        
        # Thread-safe metric storage
        self._lock = threading.Lock()
        self.performance_metrics = deque()
        self.accuracy_metrics = deque()
        self.health_metrics = deque()
        self.error_logs = deque()
        
        # Aggregated statistics
        self.component_stats = defaultdict(lambda: {
            'total_requests': 0,
            'total_duration': 0,
            'error_count': 0,
            'success_count': 0,
            'avg_response_time': 0,
            'error_rate': 0
        })
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def record_performance(self, component: str, operation: str, duration: float, 
                          success: bool, metadata: Optional[Dict] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            timestamp=time.time(),
            component=component,
            operation=operation,
            duration=duration,
            success=success,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.performance_metrics.append(metric)
            self._update_component_stats(component, duration, success)
    
    def record_medical_accuracy(self, query_type: str, entities_found: int, 
                              predictions_count: int, confidence: float, 
                              uncertainty: float, safety_flags: List[str]):
        """Record medical accuracy metrics"""
        metric = MedicalAccuracyMetric(
            timestamp=time.time(),
            query_type=query_type,
            entities_found=entities_found,
            predictions_count=predictions_count,
            confidence=confidence,
            uncertainty=uncertainty,
            safety_flags=safety_flags
        )
        
        with self._lock:
            self.accuracy_metrics.append(metric)
    
    def record_system_health(self, component: str, status: str, cpu_usage: float,
                           memory_usage: float, error_rate: float, response_time: float):
        """Record system health metrics"""
        metric = SystemHealthMetric(
            timestamp=time.time(),
            component=component,
            status=status,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            error_rate=error_rate,
            response_time=response_time
        )
        
        with self._lock:
            self.health_metrics.append(metric)
    
    def record_error(self, component: str, error_type: str, error_message: str, 
                    context: Optional[Dict] = None):
        """Record an error event"""
        error_log = {
            'timestamp': time.time(),
            'component': component,
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {}
        }
        
        with self._lock:
            self.error_logs.append(error_log)
    
    def _update_component_stats(self, component: str, duration: float, success: bool):
        """Update aggregated component statistics"""
        stats = self.component_stats[component]
        stats['total_requests'] += 1
        stats['total_duration'] += duration
        
        if success:
            stats['success_count'] += 1
        else:
            stats['error_count'] += 1
        
        stats['avg_response_time'] = stats['total_duration'] / stats['total_requests']
        stats['error_rate'] = stats['error_count'] / stats['total_requests']
    
    def get_performance_summary(self, component: Optional[str] = None, 
                               hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for time period"""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            filtered_metrics = [
                m for m in self.performance_metrics 
                if m.timestamp >= cutoff_time and (not component or m.component == component)
            ]
        
        if not filtered_metrics:
            return {'message': 'No metrics found for specified period'}
        
        durations = [m.duration for m in filtered_metrics]
        success_count = sum(1 for m in filtered_metrics if m.success)
        
        return {
            'total_requests': len(filtered_metrics),
            'success_rate': success_count / len(filtered_metrics),
            'avg_response_time': statistics.mean(durations),
            'median_response_time': statistics.median(durations),
            'p95_response_time': self._percentile(durations, 95),
            'p99_response_time': self._percentile(durations, 99),
            'min_response_time': min(durations),
            'max_response_time': max(durations)
        }
    
    def get_medical_accuracy_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get medical accuracy summary"""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            filtered_metrics = [
                m for m in self.accuracy_metrics 
                if m.timestamp >= cutoff_time
            ]
        
        if not filtered_metrics:
            return {'message': 'No accuracy metrics found for specified period'}
        
        confidences = [m.confidence for m in filtered_metrics]
        uncertainties = [m.uncertainty for m in filtered_metrics]
        entities_counts = [m.entities_found for m in filtered_metrics]
        
        # Count safety flags
        safety_flag_counts = defaultdict(int)
        for metric in filtered_metrics:
            for flag in metric.safety_flags:
                safety_flag_counts[flag] += 1
        
        return {
            'total_queries': len(filtered_metrics),
            'avg_confidence': statistics.mean(confidences),
            'avg_uncertainty': statistics.mean(uncertainties),
            'avg_entities_per_query': statistics.mean(entities_counts),
            'safety_flags_distribution': dict(safety_flag_counts),
            'high_confidence_queries': sum(1 for c in confidences if c > 0.8),
            'low_confidence_queries': sum(1 for c in confidences if c < 0.5),
            'high_uncertainty_queries': sum(1 for u in uncertainties if u > 0.7)
        }
    
    def get_error_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get error summary"""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            filtered_errors = [
                e for e in self.error_logs 
                if e['timestamp'] >= cutoff_time
            ]
        
        error_by_component = defaultdict(int)
        error_by_type = defaultdict(int)
        
        for error in filtered_errors:
            error_by_component[error['component']] += 1
            error_by_type[error['error_type']] += 1
        
        return {
            'total_errors': len(filtered_errors),
            'errors_by_component': dict(error_by_component),
            'errors_by_type': dict(error_by_type),
            'recent_errors': filtered_errors[-10:]  # Last 10 errors
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _cleanup_old_metrics(self):
        """Remove old metrics beyond retention period"""
        cutoff_time = time.time() - self.retention_seconds
        
        with self._lock:
            # Clean performance metrics
            while self.performance_metrics and self.performance_metrics[0].timestamp < cutoff_time:
                self.performance_metrics.popleft()
            
            # Clean accuracy metrics
            while self.accuracy_metrics and self.accuracy_metrics[0].timestamp < cutoff_time:
                self.accuracy_metrics.popleft()
            
            # Clean health metrics
            while self.health_metrics and self.health_metrics[0].timestamp < cutoff_time:
                self.health_metrics.popleft()
            
            # Clean error logs
            while self.error_logs and self.error_logs[0]['timestamp'] < cutoff_time:
                self.error_logs.popleft()
    
    def _start_cleanup_thread(self):
        """Start background thread for metric cleanup"""
        def cleanup_worker():
            while True:
                try:
                    self._cleanup_old_metrics()
                    time.sleep(300)  # Clean every 5 minutes
                except Exception as e:
                    logger.error(f"Metrics cleanup error: {str(e)}")
                    time.sleep(60)  # Wait 1 minute before retry
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

class PerformanceMonitor:
    """
    Context manager for monitoring function performance
    """
    
    def __init__(self, metrics_collector: MetricsCollector, component: str, operation: str):
        self.metrics_collector = metrics_collector
        self.component = component
        self.operation = operation
        self.start_time = None
        self.success = True
        self.metadata = {}
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.success = exc_type is None
        
        if exc_type:
            self.metadata['error_type'] = exc_type.__name__
            self.metadata['error_message'] = str(exc_val)
        
        self.metrics_collector.record_performance(
            self.component, self.operation, duration, self.success, self.metadata
        )
        
        if exc_type:
            self.metrics_collector.record_error(
                self.component, exc_type.__name__, str(exc_val), self.metadata
            )
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the performance record"""
        self.metadata[key] = value

class AlertingSystem:
    """
    Alerting system for medical AI pipeline monitoring
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules = []
        self.alert_handlers = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alerting rules"""
        # High error rate
        self.add_alert_rule(
            name="high_error_rate",
            condition=lambda stats: stats.get('error_rate', 0) > 0.1,
            severity="warning",
            message="Error rate above 10%"
        )
        
        # Slow response times
        self.add_alert_rule(
            name="slow_response_time",
            condition=lambda stats: stats.get('avg_response_time', 0) > 5.0,
            severity="warning",
            message="Average response time above 5 seconds"
        )
        
        # Low confidence predictions
        self.add_alert_rule(
            name="low_confidence_predictions",
            condition=lambda stats: stats.get('avg_confidence', 1) < 0.5,
            severity="warning",
            message="Average prediction confidence below 50%"
        )
        
        # High uncertainty
        self.add_alert_rule(
            name="high_uncertainty",
            condition=lambda stats: stats.get('avg_uncertainty', 0) > 0.8,
            severity="critical",
            message="Average uncertainty above 80%"
        )
    
    def add_alert_rule(self, name: str, condition: Callable, severity: str, message: str):
        """Add a custom alert rule"""
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'severity': severity,
            'message': message,
            'last_triggered': 0
        })
    
    def add_alert_handler(self, handler: Callable[[Dict], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def check_alerts(self):
        """Check all alert rules and trigger if necessary"""
        current_time = time.time()
        
        # Get current metrics
        performance_stats = self.metrics_collector.get_performance_summary(hours=1)
        accuracy_stats = self.metrics_collector.get_medical_accuracy_summary(hours=1)
        error_stats = self.metrics_collector.get_error_summary(hours=1)
        
        combined_stats = {**performance_stats, **accuracy_stats, **error_stats}
        
        for rule in self.alert_rules:
            try:
                # Check if rule should trigger
                if rule['condition'](combined_stats):
                    # Avoid spam - only trigger once per hour
                    if current_time - rule['last_triggered'] > 3600:
                        alert = {
                            'name': rule['name'],
                            'severity': rule['severity'],
                            'message': rule['message'],
                            'timestamp': current_time,
                            'stats': combined_stats
                        }
                        
                        self._trigger_alert(alert)
                        rule['last_triggered'] = current_time
                        
            except Exception as e:
                logger.error(f"Alert rule error for {rule['name']}: {str(e)}")
    
    def _trigger_alert(self, alert: Dict):
        """Trigger alert through all handlers"""
        logger.warning(f"ALERT: {alert['name']} - {alert['message']}")
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {str(e)}")

class MedicalAnalyticsDashboard:
    """
    Analytics dashboard for medical AI system insights
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data"""
        return {
            'timestamp': time.time(),
            'system_overview': self._get_system_overview(),
            'performance_metrics': self._get_performance_dashboard(),
            'medical_metrics': self._get_medical_dashboard(),
            'component_health': self._get_component_health(),
            'trending_data': self._get_trending_data()
        }
    
    def _get_system_overview(self) -> Dict[str, Any]:
        """Get high-level system overview"""
        perf_1h = self.metrics_collector.get_performance_summary(hours=1)
        perf_24h = self.metrics_collector.get_performance_summary(hours=24)
        accuracy_1h = self.metrics_collector.get_medical_accuracy_summary(hours=1)
        errors_1h = self.metrics_collector.get_error_summary(hours=1)
        
        return {
            'requests_1h': perf_1h.get('total_requests', 0),
            'requests_24h': perf_24h.get('total_requests', 0),
            'success_rate_1h': perf_1h.get('success_rate', 0),
            'avg_response_time': perf_1h.get('avg_response_time', 0),
            'avg_confidence': accuracy_1h.get('avg_confidence', 0),
            'total_errors_1h': errors_1h.get('total_errors', 0),
            'system_status': self._determine_system_status(perf_1h, accuracy_1h, errors_1h)
        }
    
    def _get_performance_dashboard(self) -> Dict[str, Any]:
        """Get performance dashboard data"""
        return {
            'response_time_distribution': self._get_response_time_distribution(),
            'throughput_trends': self._get_throughput_trends(),
            'component_performance': self._get_component_performance(),
            'error_trends': self._get_error_trends()
        }
    
    def _get_medical_dashboard(self) -> Dict[str, Any]:
        """Get medical-specific dashboard data"""
        accuracy_stats = self.metrics_collector.get_medical_accuracy_summary(hours=24)
        
        return {
            'confidence_distribution': self._get_confidence_distribution(),
            'entity_recognition_stats': self._get_entity_stats(),
            'safety_flag_trends': accuracy_stats.get('safety_flags_distribution', {}),
            'prediction_quality_trends': self._get_prediction_quality_trends()
        }
    
    def _get_component_health(self) -> Dict[str, Any]:
        """Get component health status"""
        component_stats = dict(self.metrics_collector.component_stats)
        
        health_status = {}
        for component, stats in component_stats.items():
            if stats['error_rate'] < 0.01:
                status = 'healthy'
            elif stats['error_rate'] < 0.05:
                status = 'degraded'
            else:
                status = 'unhealthy'
            
            health_status[component] = {
                'status': status,
                'error_rate': stats['error_rate'],
                'avg_response_time': stats['avg_response_time'],
                'total_requests': stats['total_requests']
            }
        
        return health_status
    
    def _determine_system_status(self, perf_stats: Dict, accuracy_stats: Dict, error_stats: Dict) -> str:
        """Determine overall system status"""
        error_rate = 1 - perf_stats.get('success_rate', 0)
        avg_confidence = accuracy_stats.get('avg_confidence', 0)
        total_errors = error_stats.get('total_errors', 0)
        
        if error_rate > 0.1 or avg_confidence < 0.4 or total_errors > 50:
            return 'unhealthy'
        elif error_rate > 0.05 or avg_confidence < 0.6 or total_errors > 20:
            return 'degraded'
        else:
            return 'healthy'
    
    def _get_response_time_distribution(self) -> Dict[str, int]:
        """Get response time distribution buckets"""
        cutoff_time = time.time() - 3600  # Last hour
        
        with self.metrics_collector._lock:
            durations = [
                m.duration for m in self.metrics_collector.performance_metrics
                if m.timestamp >= cutoff_time
            ]
        
        buckets = {'<0.5s': 0, '0.5-1s': 0, '1-2s': 0, '2-5s': 0, '>5s': 0}
        
        for duration in durations:
            if duration < 0.5:
                buckets['<0.5s'] += 1
            elif duration < 1.0:
                buckets['0.5-1s'] += 1
            elif duration < 2.0:
                buckets['1-2s'] += 1
            elif duration < 5.0:
                buckets['2-5s'] += 1
            else:
                buckets['>5s'] += 1
        
        return buckets
    
    def _get_confidence_distribution(self) -> Dict[str, int]:
        """Get confidence score distribution"""
        cutoff_time = time.time() - 3600  # Last hour
        
        with self.metrics_collector._lock:
            confidences = [
                m.confidence for m in self.metrics_collector.accuracy_metrics
                if m.timestamp >= cutoff_time
            ]
        
        buckets = {'<0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7-0.9': 0, '>0.9': 0}
        
        for confidence in confidences:
            if confidence < 0.3:
                buckets['<0.3'] += 1
            elif confidence < 0.5:
                buckets['0.3-0.5'] += 1
            elif confidence < 0.7:
                buckets['0.5-0.7'] += 1
            elif confidence < 0.9:
                buckets['0.7-0.9'] += 1
            else:
                buckets['>0.9'] += 1
        
        return buckets
    
    # Placeholder methods for additional dashboard features
    def _get_throughput_trends(self) -> List[Dict]:
        return []
    
    def _get_component_performance(self) -> Dict:
        return {}
    
    def _get_error_trends(self) -> List[Dict]:
        return []
    
    def _get_entity_stats(self) -> Dict:
        return {}
    
    def _get_prediction_quality_trends(self) -> List[Dict]:
        return []
    
    def _get_trending_data(self) -> Dict:
        return {}