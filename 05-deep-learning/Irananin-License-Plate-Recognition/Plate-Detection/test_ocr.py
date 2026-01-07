from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='fa')

img_path = './data/images/cars.jpg'
result = ocr.predict(img_path)
for line in result:
  with open('test.txt', 'a', encoding='utf-8') as f:
    f.write(line['rec_texts'][0] + '\n')