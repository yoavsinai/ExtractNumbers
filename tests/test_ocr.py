import os
import cv2
import shutil

from src.ocr_engine import NumberExtractor
from src.generate_dynamic_numbers import generate_dataset


OUTPUT_DIR = "data/synthetic_dynamic"
ERROR_DIR = os.path.join(OUTPUT_DIR, "error")
ERROR_LOG = os.path.join(OUTPUT_DIR, "errors.txt")

NUM_IMAGES = 1000

def evaluate():

    # יצירת דאטה חדשה
    data_dir = generate_dataset(
        output_dir=OUTPUT_DIR,
        num_images=NUM_IMAGES
    )

    # 🔥 יצירת תיקיית error בתוך הדאטה שנוצרה
    error_dir = os.path.join(data_dir, "error")
    os.makedirs(error_dir, exist_ok=True)

    error_log = os.path.join(data_dir, "errors.txt")

    extractor = NumberExtractor()

    total = 0
    correct = 0

    # ניקוי לוג קודם
    open(error_log, "w").close()

    for filename in sorted(os.listdir(data_dir)):

        if not filename.endswith(".png"):
            continue

        # לא לעבד שוב קבצים שכבר בתיקיית error
        if filename == "error" or filename == "errors.txt":
            continue

        true_label = filename.split("_")[0]

        path = os.path.join(data_dir, filename)
        img = cv2.imread(path)

        prediction = extractor.predict(img)

        total += 1

        is_correct = prediction is not None and str(prediction) == true_label

        if is_correct:
            correct += 1
        else:
            # רישום שגיאה
            with open(error_log, "a") as f:
                f.write(f"{filename} | TRUE: {true_label} | PRED: {prediction}\n")

            # העתקה לתיקיית error
            shutil.copy(path, os.path.join(error_dir, filename))

            print(f"{filename} | True: {true_label} | Pred: {prediction}")

    print("\n======================")
    print("Total:", total)
    print("Correct:", correct)
    print("Accuracy:", correct / total if total > 0 else 0)
    print("======================")
    
if __name__ == "__main__":
    evaluate()