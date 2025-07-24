# POST-PROCESSING

## Overview of Post-Processing:
* Loads polygon data from JSON files in the specified folder
* Filters by target labels (PV_normal, heater, pool)
* Creates binary masks for each label and extracts polygon contours
* Calculates area in m^2
* Creates visualization of the polygonization and vertices
* Returns area by several parameters (which can easily be edited)
  * Total
  * Image ID
  * Label
  * Image ID & label
  * Saves a summary CSV
 
### Visualization
* import OpenCV and Matplotlib
* Draw contours and fill in predicted regions
* Color-coded masks per label
  * "PV_normal": Green
  * "PV_heater": Red
  * "PV_pool": Blue
* Displays edges and vertices on a 320 × 320 graph

#### Basic Overview

| Script                                | Input Format       | Area Logic               | Canvas Size | Use Case                      |
| ------------------------------------- | ------------------ | ------------------------ | ----------- | ----------------------------- |
| `unet_final_area_calculation.py`      | UNet JSONs    | All polygons             | 320 × 320     | UNet prediction output     |
| `yolo_unet_final_area_calculation.py` | YOLO + UNet JSONs    | Centroid-only polygon    | 320 × 320     | YOLO + UNet prediction outpus |
| `merged_unet_calc.py`                 | Merged large JSONs | All polygons             | 12500 × 12500 | Full-image UNet prediction   |
| `reader.py`                           | .CSV  | aggregation | N/A         | understanding data at high-level         |



| Script Path                                                      | Description                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/home/prg9/post_processing/unet_final_area_calculation.py`      | Processes JSON outputs from the UNet model. Each JSON corresponds to a 320×320 tile                              |
| `/home/prg9/post_processing/yolo_unet_final_area_calculation.py` | Designed specifically for **YOLO+UNet predictions**, where each JSON represents a single object located at the center of the tile `(160, 160)`. Calculates area **only** for the centered object to avoid duplicates. Running this on full-tile JSONs will result in overestimated + inaccurate area.                 |
| `/home/prg9/post_processing/merged_unet_calc.py`                 | similar to `unet_final_area_calculation.py`, but only works on **merged JSON files** (all the data is usually separated by each tile but in this case the data is segregated by image_id not title) . Canvas size of **12500 × 12500** (original .tif file size) size. |
| `/home/prg9/post_processing/reader.py`                           | Post-processing tool that reads the CSV output from the UNet .csv and generates condensed, high-level summaries                                                   |



**🚨🚨 The main difference between `yolo_unet_final_area_calculation.py` and `unet_final_area_calculation.py` is that YOLO + UNet only calculates the area of the object located at the _centroid_ of each .json file while the UNet calculates the area of _each_ object in the .json file. In the folder with YOLO + UNet, you need to only calculate the area of the object located at (160, 160) because each object has its own .json file. If you run the UNet code on YOLO + UNet .json, then you will get data that is severely overestimating the area because there will be numerous duplicates. 🚨🚨**



