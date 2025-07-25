# Data processing

This code integrates and refines aerial imagery and annotations. It improves the data preparation pipeline developed by the Bass Connections team in 2024, available at https://github.com/slaicha/cape_town_segmentation. The final output of this process is a GPKG (GeoPackage) file, containing about 19,000 annotations corresponding to 268 annotation files and approximately 19,000 raw annotations. The quality of these annotations is subsequently assessed in the next stage, titled Annotation Checking.

----
## Step 1: Download data

To download files from `dukebox`, using BOX API is recommended

* `CapeTown_ImageIDs.xlxs`: The list of Aerial Images ID and Annotators
* `Aerial Imagery`: Cape Town, 12500 * 12500 size, 8cm/pixel (around 55GB for 2023)

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
