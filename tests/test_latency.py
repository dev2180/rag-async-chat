import time
from app.utils.latency import LatencyTracker, QueryMetrics

def test_latency_tracker_measures_correctly():
    tracker = LatencyTracker("Mock Operation")
    
    with tracker.measure():
        time.sleep(0.05) # Sleep 50ms
        
    # Python sleep is not perfectly exact, but it should be close to 50ms (e.g., >= 45ms)
    assert tracker.duration_ms >= 45.0
    assert tracker.name == "Mock Operation"

def test_query_metrics_initialization():
    metrics = QueryMetrics()
    assert metrics.query_rewrite_ms == 0.0
    assert metrics.total_ms == 0.0
    assert isinstance(metrics.timestamp, str)
    assert len(metrics.timestamp) > 10 # Should be an ISO datetime string
