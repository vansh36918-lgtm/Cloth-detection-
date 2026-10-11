import numpy as np
from collections import deque

class MeasurementSmoother:
    """
    Weighted moving average smoother with outlier rejection.
    Recent frames have higher weight so display responds faster
    but still filters out jitter and tracking glitches.
    """
    def __init__(self, history_size=20):
        self.history_size = history_size
        self.shoulder_buf   = deque(maxlen=history_size)
        self.chest_buf      = deque(maxlen=history_size)
        self.length_buf     = deque(maxlen=history_size)
        self.hip_buf        = deque(maxlen=history_size)
        self.confidence_buf = deque(maxlen=history_size)
        self.outlier_counts = {
            "shoulder": 0,
            "chest": 0,
            "length": 0,
            "hip": 0
        }

    def _weighted_mean(self, buf):
        """Apply linearly increasing weights to favour recent readings."""
        if not buf:
            return 0.0
        vals    = np.array(buf)
        weights = np.arange(1, len(vals) + 1, dtype=float)
        return float(np.average(vals, weights=weights))

    def _add_with_outlier_filter(self, buf, val, key, threshold=0.12):
        """Filters out spikes. If 5 consecutive outliers occur, resets and accepts."""
        if len(buf) >= 5:
            median = np.median(buf)
            if median > 0 and abs(val - median) / median > threshold:
                # Outlier detected
                count = self.outlier_counts.get(key, 0)
                if count < 5:
                    self.outlier_counts[key] = count + 1
                    return # Reject this frame
                else:
                    # Reset buffer to handle step-changes
                    buf.clear()
        
        self.outlier_counts[key] = 0
        buf.append(val)

    def update(self, measurements):
        if not measurements:
            return None

        # Filter and append raw measurements
        self._add_with_outlier_filter(self.shoulder_buf, measurements["shoulderWidth"], "shoulder")
        self._add_with_outlier_filter(self.chest_buf, measurements["chestWidth"], "chest")
        self._add_with_outlier_filter(self.length_buf, measurements["garmentLength"], "length")
        
        hip_val = measurements.get("hip_width_cm", measurements["chestWidth"])
        self._add_with_outlier_filter(self.hip_buf, hip_val, "hip")
        
        # Confidence can just be appended normally
        self.confidence_buf.append(measurements["confidence"])

        # Return smoothed values if we have enough history to start
        if not self.shoulder_buf:
            return None

        return {
            "shoulderWidth": self._weighted_mean(self.shoulder_buf),
            "chestWidth":    self._weighted_mean(self.chest_buf),
            "garmentLength": self._weighted_mean(self.length_buf),
            "hip_width_cm":  self._weighted_mean(self.hip_buf),
            "confidence":    self._weighted_mean(self.confidence_buf),
        }
