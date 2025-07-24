# Data generation

This code integrates and refines aerial imagery and annotations for computer vision modeling(U-Net, Yolo). It improves the data preparation pipeline originally developed by the Bass Connections team in 2024, available at https://github.com/slaicha/cape_town_segmentation. The final output of this process is a GPKG (GeoPackage) file, containing approximately 19,000 annotations corresponding to 268 images. The quality of these annotations is assessed in the next stage.

----
## Step 1: Download data

To download files from `dukebox`, using BOX API is recommended

* `CapeTown_ImageIDs.xlxs`: Aerial Imagery ID와 Annotator들의 목록
* `Aerial Imagery`: CapeTown의 항공사진 (12500 * 12500 size, 8cm/pixel)

## Step 2: Pre-processing

Load files and clean the data, unify format

* `CRS`: Coordinate Reference System = ESRI:102562
* `Annotator Lists`: Contains a list of annotated images
* `Shapefile`: List of anotations

## Step 3: Bounding Box extraction

* Extract vertices of each aerial image directly from files
* Match annoatations with aerial imagery
* Calculate the area of each polygon using its geometry

## Step 4: Annotation Post-processing

* Drop duplicated annotations correponsing to 'geometry' 
* Merge PV_Pool into PV_pool
* Ensure binary values '1' or 'Nan' for PV related columns
* Create a 'PV_normal' column if all other PV related colums are NaN
* Reorder columns for clarity
* Drop unnecessary columns (layer, path = '2020 layers')
* Reindex 'id' for consistency

## Step 5: Save Dataframe as GPKG

* NOTE: ESRI is not directly readbable in python. Therefore it's safe to use WKT format instead.
