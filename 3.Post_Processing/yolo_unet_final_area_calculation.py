import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import Point, Polygon
from collections import defaultdict
import csv

# === Configuration ===
json_folder = '/shared/data/climateplus2025/Prediction_EntireDataset/U-Net_prediction_output/prediction_outputs_v44'
csv_path = "/home/prg9/post_processing/FINAL_yolo_unet_summary.csv"

# Target labels and colors
target_labels = {
    "PV_normal": (0, 255, 0),
    "PV_heater": (255, 0, 0),
    "PV_pool": (0, 0, 255)
}

canvas_height, canvas_width = 320, 320
center_point = (160, 160)
PIXEL_AREA_M2 = 0.08 ** 2  # 8cm per pixel

# === Count JSON files ===
json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
print(f"\n📦 Found {len(json_files)} JSON files in '{json_folder}'")

# === Accumulators ===
total_area_m2 = 0.0
area_by_image_id = defaultdict(float)
area_by_label = defaultdict(float)
area_by_image_and_label = defaultdict(lambda: defaultdict(float))

# === Main Loop ===
for filename in json_files:
    filepath = os.path.join(json_folder, filename)
    with open(filepath, 'r') as f:
        data = json.load(f)

    pred_coords = data.get("predicted_coords", {})
    active_labels = [label for label in pred_coords if label in target_labels]

    if not active_labels:
        continue

    image_id = filename.split('_tile_')[0].replace("i_", "")
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    found_label = None
    found_area_px = None
    found_area_m2 = None

    print(f"\n📁 {filename} → Labels: {', '.join(active_labels)}")

    for label in active_labels:
        coords = pred_coords[label]
        color = target_labels[label]

        mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        for row, col in coords:
            if 0 <= row < canvas_height and 0 <= col < canvas_width:
                canvas[row, col] = color
                mask[row, col] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours):
            if len(cnt) < 3:
                continue

            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            polygon_pts = [tuple(pt[0]) for pt in approx]

            cv2.polylines(canvas, [approx], isClosed=True, color=(255, 255, 0), thickness=1)

            print(f"\n  🔹 {label} [Object {i+1}] Vertices:")
            for j, pt in enumerate(approx):
                x, y = pt[0]
                print(f"    Point {j+1}: (x={x}, y={y})")
                cv2.circle(canvas, (x, y), 2, (255, 105, 180), -1)

            poly = Polygon(polygon_pts)
            if poly.contains(Point(center_point)) and not found_label:
                found_label = label

                # ✅ Use OpenCV-based area to avoid overestimation
                found_area_px = cv2.contourArea(cnt)
                found_area_m2 = found_area_px * PIXEL_AREA_M2

                # Update accumulators
                total_area_m2 += found_area_m2
                area_by_image_id[image_id] += found_area_m2
                area_by_label[label] += found_area_m2
                area_by_image_and_label[image_id][label] += found_area_m2
                break

        if found_label:
            break

    # Draw center marker
    cv2.drawMarker(canvas, center_point, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

    if found_label:
        print(f"\n📍 Center (160,160) is inside '{found_label}' in {filename}")
        print(f"\n📐 Area of center polygon:")
        print(f"    {found_area_px:.2f} pixels²")
        print(f"    {found_area_m2:.2f} m²")
    else:
        print(f"\n⚠️ Center (160,160) not inside any target object in {filename}")

    # Show visual
    plt.figure(figsize=(4, 4))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title(f"{filename}")
    plt.axis('off')
    plt.show()

# === Final Summary ===
print("\n🧮 Total Area Summary")
print(f"📐 Total center-object area: {total_area_m2:.2f} m²")

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

    # Total
    writer.writerow(["total", "all_labels", f"{total_area_m2:.2f}"])

    # Per label
    for label, area in sorted(area_by_label.items()):
        writer.writerow(["total_by_label", label, f"{area:.2f}"])

    # Per image total
    for image_id, area in sorted(area_by_image_id.items()):
        writer.writerow(["total_by_image", image_id, f"{area:.2f}"])

    # PV_normal per image
    for image_id, label_areas in sorted(area_by_image_and_label.items()):
        pv_area = label_areas.get("PV_normal", 0.0)
        writer.writerow(["pv_normal_by_image", image_id, f"{pv_area:.2f}"])

print(f"\n✅ CSV saved to: {csv_path}")
