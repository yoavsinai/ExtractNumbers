import cv2
import sys
from src.ocr_engine import NumberExtractor


def main(image_path):
    if not image_path:
        print("Please provide image path")
        return

    image = cv2.imread(image_path)

    if image is None:
        print("Failed to read image")
        return

    extractor = NumberExtractor()

    result = extractor.predict(image)

    print("Predicted number:", result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ocr_on_image.py path/to/image.jpg")
    else:
        main(sys.argv[1])