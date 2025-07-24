import csv
from collections import defaultdict

# === CONFIG ===
input_csv = "/home/prg9/post_processing/summary_unet_all_polygons.csv"
output_csv = "/home/prg9/post_processing/grouped_summary.csv"

# === Accumulators ===
total_by_image = defaultdict(float)
pv_normal_by_image = defaultdict(float)

# === Helper to extract base image ID
def get_base_image_id(filename):
    parts = filename.replace(".json", "").split("_")
    return "_".join(parts[:4])  # e.g., '2023_RGB_8cm_W24A_17'

# === Read input file ===
with open(input_csv, "r", newline="") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        metric = row["Metric"]
        identifier = row["Identifier"]
        try:
            area_m2 = float(row["Area_m2"])
        except ValueError:
            continue  # Skip invalid rows

        if metric == "total_by_image":
            base_image = get_base_image_id(identifier)
            total_by_image[base_image] += area_m2

        elif metric == "pv_normal_by_image":
            base_image = get_base_image_id(identifier)
            pv_normal_by_image[base_image] += area_m2

# === Write grouped output ===
with open(output_csv, "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Metric", "Identifier", "Area_m2"])

    for image_id, area in sorted(total_by_image.items()):
        writer.writerow(["total_by_image", image_id, f"{area:.2f}"])

    for image_id, area in sorted(pv_normal_by_image.items()):
        writer.writerow(["pv_normal_by_image", image_id, f"{area:.2f}"])

print(f"✅ Grouped summary saved to: {output_csv}")
