ReadMe.md for Post-Processing

Overall purpose of the main post-processing is that 
* Loads polygon data from JSON files in the specified folder
* Filters by target labels (PV_normal, heater, pool)
* Creates binary masks for each label and extracts polygon contours
* Calculates area in m^2
* Creates visualization of the polygonization and where the vertices are located
* Returns area by several parameters (which can easily be edited)
  * Total
  * Image ID
  * Label
  * Image ID & label
  * Saves a summary CSV

** 🚨🚨 The main difference between `yolo_unet_final_area_calculation.py` and `unet_final_area_calculation.py` is that YOLO + UNet only calculates the area of the object located at the _centroid_ of each .json file while the UNet calculates the area of _each_ object in the .json file. In the folder with YOLO + UNet, you need to only calculate the area of the object located at (160, 160) because each object has its own .json file. If you run the UNet code on YOLO + UNet .json, then you will get data that is severely overestimating the area because there will be numerous duplicates. 🚨🚨 **

`/home/prg9/post_processing/yolo_unet_final_area_calculation.py`


`/home/prg9/post_processing/unet_final_area_calculation.py` & `/home/prg9/post_processing/reader.py`
The `reader.py` purpose is to process and condense the output .csv file from the unet_final_area_calculation.py to get more high level summaries and results rather than more specific and detailed ones.

`/home/prg9/post_processing/merged_unet_calc.py` 
This file has the same purpose as the `unet_final_area_calculation.py` file. The only difference is that the `merged_unet_calc.py` file runs on a different format of a .json file which combines all the tiles of the image into one file. Additionally, the bounds for the `merged_unet_calc.py` is 12500 x 12500 because that is the size of the .tif file while for unet_final_area_calculation.py the bounds are 320 x 320 because that is the UNet model’s input size.
