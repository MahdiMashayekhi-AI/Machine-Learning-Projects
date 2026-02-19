import cv2


class StreamProcessor:
  def __init__(self, pipeline):
    self.pipeline = pipeline


  def run(self, source):
    cap = cv2.VideoCapture(source)

    while True:
      ret, frame = cap.read()
      if not ret:
        break

      result = self.pipeline.process_frame(frame)
      yield result