from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DetectionResult:
  bbox: Tuple[int, int, int, int]
  conf: float
  class_id: int | None