"""Extract placement-suitability (intent) boxes from intent maps into a CSV.

Each map in ``--map-dir`` is thresholded at ``--threshold`` (fraction of 255),
cleaned with a small morphological opening/closing, and every connected region
covering at least ``--min-area`` of the canvas becomes one box (largest first,
at most ``--max-boxes``). If no region survives, the map is thresholded at its
own mean instead, so every image contributes at least one box.

Output format matches the ``*_intent_mbbox.csv`` files read by the dataloader:
one row per box with columns ``poster_path`` and ``box_elem`` = "(x1, y1, x2, y2)"
in pixel coordinates of the map.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd


def regions(intent_map: np.ndarray, threshold: float, kernel: int, min_area: float,
            max_boxes: int) -> list[tuple[int, int, int, int]]:
    canvas = float(intent_map.shape[0] * intent_map.shape[1])
    _, binary = cv2.threshold(intent_map, threshold, 255, cv2.THRESH_BINARY)
    if kernel > 1:
        ones = np.ones((kernel, kernel), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ones)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ones)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= min_area * canvas:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes[:max_boxes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--min-area", type=float, default=0.002)
    parser.add_argument("--max-boxes", type=int, default=9)
    args = parser.parse_args()

    rows, fallbacks = [], 0
    names = sorted(n for n in os.listdir(args.map_dir) if n.lower().endswith((".png", ".jpg")))
    for name in names:
        intent_map = cv2.imread(os.path.join(args.map_dir, name), cv2.IMREAD_GRAYSCALE)
        boxes = regions(intent_map, args.threshold * 255, args.kernel, args.min_area, args.max_boxes)
        if not boxes:
            fallbacks += 1
            boxes = regions(intent_map, float(intent_map.mean()), args.kernel, 0.0, 1) or [
                (0, 0, intent_map.shape[1], intent_map.shape[0])
            ]
        rows += [{"poster_path": name, "box_elem": str(box)} for box in boxes]
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"{args.output}: {len(rows)} boxes for {len(names)} maps ({fallbacks} mean-threshold fallbacks)")


if __name__ == "__main__":
    main()
