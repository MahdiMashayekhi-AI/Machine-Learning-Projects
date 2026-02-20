import re
from core.entities.ocr import OCRResult


class PlateValidator:
  def __init__(self):
    self._pattern = re.compile(r"^[1-9][0-9][ابپتثجدزژسشصطعقلمنوهی][1-9][0-9]{2}[1-9][0-9]$")

  
  def normalize(self, text):
    results = ""
    
    text = text.strip()

    for t in text:
      t = t.replace(" ", "")
      t = t.replace("_", "")

      results += t

    return results


  def validate(self, ocr_result: OCRResult):
    clean_text = self.normalize(ocr_result.text)

    return self._pattern.match(clean_text)