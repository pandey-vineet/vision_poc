import cv2
import pytesseract
import re
from collections import defaultdict

img = cv2.imread("t_1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Preprocess to highlight text/label regions ---
# 1) Denoise slightly
gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

# 2) Morphological gradient highlights edges (good for text)
grad = cv2.morphologyEx(
    gray_blur, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
)

# 3) Binarize
bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# 4) Close gaps so letters form blobs (tune kernel if needed)
closed = cv2.morphologyEx(
    bw,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)),
    iterations=2,
)

# --- Find candidate regions ---
cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

H, W = gray.shape[:2]
candidates = []

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)

    # Filter out tiny or huge regions
    area = w * h
    if area < 0.002 * (W * H):  # too small
        continue
    if area > 0.60 * (W * H):  # too big
        continue

    # Prefer wide-ish regions (labels tend to be wide)
    aspect = w / float(h)
    if aspect < 1.2:
        continue

    candidates.append((x, y, w, h))

# Sort candidates largest-first (label often among top)
candidates.sort(key=lambda r: r[2] * r[3], reverse=True)


def normalize_token(t: str) -> str:
    return t.replace("—", "-").replace("–", "-").replace("−", "-").strip()


# Token filters for reconstruction
TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-]{1,}$")  # >=2 chars
PARTLIKE_RE = re.compile(r"^[A-Z0-9\-]{6,}$")  # part-number-ish

CONF_THRESHOLD = 55


def ocr_region(roi, tag=""):
    # ROI preprocessing tuned for text
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # OCR (two configs: general + whitelist)
    configs = [
        ("general", "--oem 3 --psm 6"),
        (
            "whitelist",
            "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-",
        ),
    ]

    all_recons = []

    for name, config in configs:
        data = pytesseract.image_to_data(
            b, output_type=pytesseract.Output.DICT, config=config
        )

        print(f"\n=== REGION {tag} | PASS: {name} ===")
        lines = defaultdict(list)

        for i in range(len(data["text"])):
            text = normalize_token(data["text"][i])
            try:
                conf = int(float(data["conf"][i]))
            except Exception:
                conf = -1

            # Print everything
            if text:
                print(f"{text} (confidence={conf})")

            if not text:
                continue

            is_token = bool(TOKEN_RE.match(text))
            is_partlike = bool(PARTLIKE_RE.match(text.upper()))
            keep = (conf >= CONF_THRESHOLD and is_token) or (is_partlike and is_token)

            if keep:
                left = data["left"][i]
                block = data["block_num"][i]
                par = data["par_num"][i]
                line = data["line_num"][i]
                lines[(block, par, line)].append((left, text))

        reconstructed = []
        for key in sorted(lines.keys()):
            tokens = [t for _, t in sorted(lines[key], key=lambda x: x[0])]
            reconstructed.append(" ".join(tokens))

        print("\n--- Reconstructed (filtered) ---")
        print("\n".join(reconstructed))
        all_recons.extend(reconstructed)

    # Score reconstructions: reward label-like tokens
    def score(lines):
        s = 0
        joined = " ".join(lines).upper()
        # Heuristics: typical RAM label patterns
        for needle in ["SK", "HYNIX", "PC4", "GB", "RX", "HMA"]:
            if needle in joined:
                s += 3
        # Reward length (more useful text)
        s += min(len(joined) // 15, 10)
        return s

    # Return best set of lines from both passes
    best = sorted([all_recons], key=lambda l: score(l), reverse=True)[0]
    return best


# OCR top N candidate regions (keeps it fast)
TOP_N = min(5, len(candidates))
best_overall = []
best_score = -1


def overall_score(lines):
    joined = " ".join(lines).upper()
    s = 0
    for needle in ["SK", "HYNIX", "PC4", "GB", "RX", "HMA", "KOREA"]:
        if needle in joined:
            s += 4
    s += min(len(joined) // 15, 10)
    return s


for idx, (x, y, w, h) in enumerate(candidates[:TOP_N], start=1):
    # Add padding so we don't cut off characters
    pad = 8
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)

    roi = img[y0:y1, x0:x1]
    lines = ocr_region(roi, tag=f"{idx} ({x0},{y0},{x1 - x0},{y1 - y0})")

    s = overall_score(lines)
    if s > best_score:
        best_score = s
        best_overall = lines

print("\n=== BEST OVERALL RECONSTRUCTION ===")
print("\n".join(best_overall))
