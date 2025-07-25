# Annotation checking and confusion matrix 

<img src="https://github.com/user-attachments/assets/223e0ddd-6bde-4741-a392-79cc586298c2" alt="설명" width="700"/>

-----
This code is a GUI-based program designed for quality checking of pre-generated annotations. It is intended to run locally and is not compatible with Pizer environments.
The program loads an annotation file in GPKG format, and for each reviewed annotation, it records the result by creating a new column name

**Note:**
1) When clicking either `resizing` or `uncertain`, the user must click one additional button (e.g., PV_normal, PV_pool, or PV_heater)
2) When `uncertain` is selected, the corresponding image is saved. So it can be reviewed seperately later.
3) The heater mat was decided to be classified as a type of water heater. However, this code still includes PV_heater_mat class, which can be removed
   
-----

## Step 1 Set the input data 

Before launching the tool, ensure the following:

1. GeoPackage (.gpkg) file containing annotation geometries and metadata (with columns like image_name, PV_normal, PV_heater, etc.)
2. A folder containing the corresponding .tif images.

##  Step 2 Launch the QC Checker GUI

* Run `checker_final_exe.ipynb`

* if using the notebook version:
  1. Open Annotation_QC_Checker_GUI.ipynb
  2. Run all cells from top to bottom

The GUI will launch and allow you to:
* View original and annotated image tiles → Apply QC labels via buttons → Zoom in/out on image tiles → Automatically save labels to the `.gpkg` file
* You can inspect the `.gpkg` file using DB Browser for SQLite.
* If any mislabeling is found, open the file in QGIS and manually correct the label based on its unique ID.

## Label Name and its discription 
* PV_normal_qc → Correct annotation for a PV panel (Glossy, uniform, and neatly arranged in rows)
* PV_heater_qc → Solar water heater detected ( Small square-like and has a White rectangular tank attached to it)
* PV_pool_qc → Solar pool heating system (Typically located next to a pool and in a darker shade, Sometimes there is visible piping nearby)
* uncertflag_qc → Not confident about the annotation
* delete_qc → Mark for deletion
* resizing_qc → Annotation needs resizing
* PV_heater_mat → a type of tankless water heater mat (This type should merged into PV_heater)







