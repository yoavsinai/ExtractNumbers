# Video Evaluation Pipeline — Problems & Solutions

---

## Problem 1 — MNIST Labels Not Filtered Correctly (`"label": "-1"`)

### What is happening
The conversion script writes `-1` as a **string** (`"-1"`) instead of an integer (`-1`).  
In `parse_anno_data` the check is:
```python
UNKNOWN_LABELS = {"digit", "digit_seq", None, -1, ""}
```
The check `label in UNKNOWN_LABELS` misses `"-1"` (string) because `-1` (int) ≠ `"-1"` (string).  
As a result, MNIST digits pass into `digit_info` with a meaningless label, polluting the accuracy metrics.

### Solution
**In the conversion script** — write `-1` as int:
```python
"label": -1,  # int, not "-1"
```

**In `parse_anno_data`** — add the string version to the set:
```python
UNKNOWN_LABELS = {"digit", "digit_seq", None, -1, "", "-1"}  # added "-1" as string
```

---

### תרגום — בעיה 1 — תוויות MNIST לא מסוננות נכון

#### מה קורה
סקריפט ההמרה כותב `-1` כ**מחרוזת** (`"-1"`) במקום כמספר שלם (`-1`).  
הבדיקה `label in UNKNOWN_LABELS` מחמיצה את `"-1"` כי `-1` (int) ≠ `"-1"` (string).  
כתוצאה, ספרות MNIST עוברות לתוך `digit_info` עם תווית חסרת משמעות ומזהמות את מדדי הדיוק.

#### פתרון
בסקריפט ההמרה — לכתוב `-1` כ-int.  
ב-`parse_anno_data` — להוסיף `"-1"` כ-string לסט `UNKNOWN_LABELS`.

---
---

## Problem 2 — Inference Resolution Too Small for DSText Videos

### What is happening
The model runs with `imgsz=256` on frames that are **1920×1080**.  
When the image is downscaled, small text boxes (e.g. `width: 16px, height: 22px`) become roughly **2 pixels wide** — the model simply cannot see them.  
This explains:
```
S1 IoU (DSText):  0.0085
S1 IoU (BOVText): 0.0000
S1 IoU (ICDAR):   0.0000
```

### Solution
Adapt `imgsz` dynamically based on frame size:
```python
h, w = img.shape[:2]
infer_size = 256 if max(h, w) <= 256 else 1280
res = model.predict(source=img, imgsz=infer_size, verbose=False)
```
Apply this change in all three eval files:
- `eval_video_global_bbox.py`
- `eval_video_individual_bbox.py`
- `eval_video_pipeline.py`

---

### תרגום — בעיה 2 — רזולוציית Inference קטנה מדי

#### מה קורה
המודל רץ עם `imgsz=256` על פריימים בגודל 1920×1080.  
כאשר התמונה מוקטנת, תיבות טקסט קטנות (לדוגמה רוחב 16 פיקסל) הופכות לכ-2 פיקסל — המודל פשוט לא רואה אותן.  
זה מסביר את ה-IoU של 0.0 ב-DSText, BOVText ו-ICDAR.

#### פתרון
להתאים את `imgsz` דינמית לפי גודל הפריים — 256 לסרטוני MNIST הקטנים, 1280 לסרטונים בגדולים.  
לשנות זאת בשלושת קבצי ה-eval.

---
---

## Problem 3 — Evaluation Includes Non-Numeric Annotations

### What is happening
DSText annotations include non-numeric text such as:
```
"SAYEED PLAY YT", "SHOP", "BRAWLERS", "SOLO SHOWDOWN" ...
```
The model was trained to detect **numbers only**, but the eval computes IoU against **all** boxes including plain text.  
Non-numeric boxes will always be misses — artificially lowering Recall and IoU.

### Solution
Filter to numeric-only boxes before evaluation:
```python
def is_numeric_value(number_entry):
    val = number_entry.get('full_value', '')
    cleaned = ''.join(c for c in str(val) if c.isdigit())
    return len(cleaned) > 0

# In the frame loop — filter before building global_boxes
numeric_numbers = [n for n in frame_data.get('detected_numbers', [])
                   if is_numeric_value(n)]

if not numeric_numbers:
    continue
```
Apply this change in all three eval files:
- `eval_video_global_bbox.py`
- `eval_video_individual_bbox.py`
- `eval_video_pipeline.py`

---

### תרגום — בעיה 3 — הערכה על אנוטציות לא מספריות

#### מה קורה
אנוטציות DSText כוללות ערכים כמו "SHOP", "BRAWLERS", "SOLO SHOWDOWN" וכדומה.  
המודל אומן לזהות מספרים בלבד, אבל ה-eval מחשב IoU מול כל התיבות כולל טקסט רגיל.  
תיבות לא מספריות תמיד יהיו miss — מה שמוריד את ה-Recall וה-IoU באופן מלאכותי.

#### פתרון
לסנן לפני ההערכה רק תיבות עם ערך מספרי.  
לשנות זאת בשלושת קבצי ה-eval.

---
---

## Summary of Expected Impact / סיכום השפעות צפויות

| Dataset | Main Problem | S1 IoU Now | Expected After Fix |
|---|---|---|---|
| moving_mnist | `-1` as string label | 0.5656 ✓ | bbox unchanged, digit acc correctly zeroed |
| dstext_v2 | small imgsz + non-numeric boxes | 0.0085 | significant improvement |
| bovtext | small imgsz | 0.0000 | significant improvement |
| icdar_svt | small imgsz | 0.0000 | significant improvement |

---

## Recommended Order of Actions / סדר פעולות מומלץ

1. Fix `parse_anno_data` — add `"-1"` to `UNKNOWN_LABELS`
2. Fix the conversion script — write `"label": -1` as int
3. **Re-run the conversion script** to regenerate JSON files with correct labels
4. Fix dynamic `imgsz` in all three eval files
5. Add numeric filtering in all three eval files
6. Re-run the evaluation