import os
import torch
from ultralytics import YOLO

if torch.cuda.is_available():
    print("cuda is available")
    device = 0
else:
    print("cuda is not available")
    device = 'cpu'


def main():
    model = YOLO('yolo11n.pt')

    print('Starting training...')

    results = model.train(
        data='./dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        workers=0,
        name='lpd_model_v2',
        patience=20,
        save=True,
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True
    )

    print('Training complete.')

    print("Evaluating model...")
    final_model_path = os.path.join(
        'runs', 'detect', 'lpd_model_v2', 'weights', 'best.pt')
    best_model = YOLO(final_model_path)
    metrics = best_model.val()

    print(f"Final mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"Final mAP@0.5:0.95: {metrics.box.map:.4f}")

    print("Exporting to ONNX...")
    success = model.export(format='onnx')

    if success:
        print("ONNX export successful")
    else:
        print("ONNX export failed")


if __name__ == "__main__":
    main()
