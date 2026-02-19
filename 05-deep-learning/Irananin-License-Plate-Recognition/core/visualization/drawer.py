import cv2

class Drawer:
    def draw(self, frame, results):
        for r in results:
            x1, y1, x2, y2 = r.bbox

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            label = f"{r.text} ({r.det_conf:.2f})"

            cv2.putText(
                frame,
                label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )
        return frame