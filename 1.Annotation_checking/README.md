README.md



# Annotation cheecking and confution matrix 

## Step 1 set the input data 

Before launching the tool, ensure the following:

1. You have a GeoPackage (.gpkg) file containing annotation geometries and metadata (with columns like image_name, PV_normal, PV_heater, etc.)
2. You have a folder containing the corresponding .tif images.

##  Step 2 Launch the QC Checker GUI

if using the notebook version:
1. Open Annotation_QC_Checker_GUI.ipynb
2. Run all cells from top to bottom

The GUI will launch and allow you to:
View original and annotated image tiles--> Apply QC labels via buttons --> Zoom in/out on image tiles -- >Automatically save labels to the .gpkg file. You can inspect the data and labels using DB Browser for SQLite. If any mislabeling is found, open the GeoPackage file in QGIS and manually correct the label based on its unique ID.









