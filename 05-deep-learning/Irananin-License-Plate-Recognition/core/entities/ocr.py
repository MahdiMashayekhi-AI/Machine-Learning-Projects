from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OCRResult:
  text: str
  conf: Optional[float] = None