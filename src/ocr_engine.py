import cv2
from paddleocr import PaddleOCR


class NumberExtractor:
    def __init__(self):
        self.reader = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            # use_gpu=True
        )

    def predict(self, image):
            results = self.reader.ocr(image)

            if not results or results[0] is None:
                return None

            for line in results[0]:
                bbox, (text, confidence) = line

                digits = ''.join(filter(str.isdigit, text))
                if digits:
                    return int(digits[0])

            return None