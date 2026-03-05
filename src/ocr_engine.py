import cv2
from paddleocr import PaddleOCR


class NumberExtractor:
    def __init__(self):
        self.reader = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False  # כרגע CPU
        )

    def predict(self, image):
        results = self.reader.ocr(image)

        if not results or results[0] is None:
            return None

        digits_list = []

        for line in results[0]:
            bbox, (text, confidence) = line

            digits = ''.join(filter(str.isdigit, text))

            if digits:
                digits_list.append((bbox, digits))

        if not digits_list:
            return None

        # מיון משמאל לימין לפי x של הבוקס
        digits_list.sort(key=lambda item: item[0][0][0])
        
        full_number = ''.join([d[1] for d in digits_list])

        return (full_number)