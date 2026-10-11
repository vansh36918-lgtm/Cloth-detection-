import numpy as np
from collections import deque

class MeasurementSmoother:
    """
    Weighted moving average smoother.
    Recent frames have higher weight so display responds faster
    but still filters out jitter.
    """
    def __init__(self, history_size=20):
        self.history_size = history_size
        self.shoulder_buf   = deque(maxlen=history_size)
        self.chest_buf      = deque(maxlen=history_size)
        self.length_buf     = deque(maxlen=history_size)
        self.hip_buf        = deque(maxlen=history_size)
        self.confidence_buf = deque(maxlen=history_size)

    def _weighted_mean(self, buf):
        """Apply linearly increasing weights to favour recent readings."""
        vals    = np.array(buf)
        weights = np.arange(1, len(vals) + 1, dtype=float)
        return float(np.average(vals, weights=weights))

    def update(self, measurements):
        if not measurements:
            return None

        self.shoulder_buf.append(measurements["shoulderWidth"])
        self.chest_buf.append(measurements["chestWidth"])
        self.length_buf.append(measurements["garmentLength"])
        self.hip_buf.append(measurements.get("hip_width_cm", measurements["chestWidth"]))
        self.confidence_buf.append(measurements["confidence"])

        return {
            "shoulderWidth": self._weighted_mean(self.shoulder_buf),
            "chestWidth":    self._weighted_mean(self.chest_buf),
            "garmentLength": self._weighted_mean(self.length_buf),
            "hip_width_cm":  self._weighted_mean(self.hip_buf),
            "confidence":    self._weighted_mean(self.confidence_buf),
        }
