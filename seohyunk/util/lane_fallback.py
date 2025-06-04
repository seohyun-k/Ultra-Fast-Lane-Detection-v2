# utils/lane_fallback.py

from collections import deque
import numpy as np

class LaneHistoryBuffer:
    """
    차선 인식 실패 시 과거 차선 정보로 보완하는 버퍼
    """
    def __init__(self, max_length=5):
        self.buffer = deque(maxlen=max_length)

    def update(self, lanes):
        if lanes:
            self.buffer.append(lanes)

    def get_latest(self):
        if self.buffer:
            return self.buffer[-1]
        else:
            return []

    def is_empty(self):
        return len(self.buffer) == 0


class SteeringFallbackController:
    """
    조향값 유지/점진 복귀 로직
    """
    def __init__(self, max_missing=5, decay_rate=0.8):
        self.missing_count = 0
        self.max_missing = max_missing
        self.decay_rate = decay_rate
        self.last_steer = 0.0

    def update(self, steer_angle, detected=True):
        """
        인식 성공/실패 여부에 따라 조향값 결정
        """
        if detected:
            self.missing_count = 0
            self.last_steer = steer_angle
            return steer_angle
        else:
            self.missing_count += 1
            if self.missing_count < self.max_missing:
                return self.last_steer
            else:
                self.last_steer *= self.decay_rate
                return self.last_steer

