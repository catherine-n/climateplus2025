import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import Polygon
from collections import defaultdict
import csv

# === Configuration ===
json_folder = '/shared/data/climateplus2025/YOLO+U-Net_Prediction_updated_0722/Merged_prediction_Json/unified_json'
csv_path = "/home/prg9/post_processing/summary_unet_all_polygons.csv"

target_labels = {
    "PV_normal": (0, 255, 0),
    "PV_heater": (255, 0, 0),
    "PV_pool": (0, 0, 255)
}

PIXEL_AREA_M2 = 0.08 ** 2  # 8cm per pixel

# === Accumulators ===
total_area_m2 = 0.0
area_by_image_id = defaultdict(float)
area_by_label = defaultdict(float)
area_by_image_and_label = defaultdict(lambda: defaultdict(float))

# === List all merged .json files ===
json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
print(f"\n📦 Found {len(json_files)} merged JSON files in '{json_folder}'")

for filename in json_files:
    filepath = os.path.join(json_folder, filename)
    with open(filepath, 'r') as f:
        data = json.load(f)

    pred_coords = data.get("predicted_coords", {})
    active_labels = [label for label in pred_coords if label in target_labels]
    if not active_labels:
        continue

    # Combine all coords to get canvas bounds
    all_coords = [pt for label_pts in pred_coords.values() for pt in label_pts]
    if not all_coords:
        continue

    max_row = max(pt[0] for pt in all_coords) + 1
    max_col = max(pt[1] for pt in all_coords) + 1

    canvas_height, canvas_width = max_row, max_col
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    image_id = os.path.splitext(filename)[0]
    print(f"\n📁 {filename} → Labels: {', '.join(active_labels)}")
    print(f"  🔍 Coord bounds → rows: 0–{max_row-1}, cols: 0–{max_col-1}")

    for label in active_labels:
        coords = pred_coords[label]
        color = target_labels[label]

        mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        skipped = 0
        for row, col in coords:
            if 0 <= row < canvas_height and 0 <= col < canvas_width:
                mask[row, col] = 255
                canvas[row, col] = color
            else:
                skipped += 1

        if skipped > 0:
            print(f"    ⚠ Skipped {skipped} out-of-bounds pixels for label '{label}'")

        # Extract contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for i, cnt in enumerate(contours):
            if len(cnt) < 3:
                continue

            polygon_pts = [tuple(pt[0]) for pt in cnt]
            poly = Polygon(polygon_pts)

            if not poly.is_valid or poly.area == 0:
                continue

            area_px = poly.area
            area_m2 = area_px * PIXEL_AREA_M2

            # Accumulate
            total_area_m2 += area_m2
            area_by_image_id[image_id] += area_m2
            area_by_label[label] += area_m2
            area_by_image_and_label[image_id][label] += area_m2

            # Draw for visualization
            cv2.polylines(canvas, [np.array(polygon_pts, dtype=np.int32)], isClosed=True, color=(255, 255, 0), thickness=1)
            for pt in polygon_pts:
                cv2.circle(canvas, pt, 1, (255, 105, 180), -1)

            print(f"\n  🔹 {label} [Object {i+1}]")
            print(f"    Area: {area_px:.2f} px² → {area_m2:.2f} m²")

    # Visualize
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title(f"{filename}")
    plt.axis('off')
    plt.show()

# === Final Summary ===
print("\n🧮 Total Area Summary")
print(f"📐 Total area (all polygons): {total_area_m2:.2f} m²")

print("\n📊 Area by image_id:")
for image_id, area in sorted(area_by_image_id.items()):
    print(f"  {image_id}: {area:.2f} m²")

print("\n📊 Area by label:")
for label, area in sorted(area_by_label.items()):
    print(f"  {label}: {area:.2f} m²")

print("\n📊 Area by image_id for PV_normal only:")
for image_id, label_areas in sorted(area_by_image_and_label.items()):
    pv_area = label_areas.get("PV_normal", 0.0)
    print(f"  {image_id}: {pv_area:.2f} m²")

# === Save CSV Summary ===
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Metric", "Identifier", "Area_m2"])

    writer.writerow(["total", "all_labels", f"{total_area_m2:.2f}"])

    for label, area in sorted(area_by_label.items()):
        writer.writerow(["total_by_label", label, f"{area:.2f}"])

    for image_id, area in sorted(area_by_image_id.items()):
        writer.writerow(["total_by_image", image_id, f"{area:.2f}"])

    for image_id, label_areas in sorted(area_by_image_and_label.items()):
        pv_area = label_areas.get("PV_normal", 0.0)
        writer.writerow(["pv_normal_by_image", image_id, f"{pv_area:.2f}"])

print(f"\n✅ CSV saved to: {csv_path}")