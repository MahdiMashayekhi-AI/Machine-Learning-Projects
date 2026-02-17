from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class PipelineResult:
  bbox: Tuple[int, int, int, int]
  text: str
  det_conf: float
  ocr_conf: Optional[float] = None
