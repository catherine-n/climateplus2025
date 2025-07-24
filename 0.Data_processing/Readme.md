# Data generation
Contributor: Shawn Lee

이 코드는 computer vision modeling에 필요한 Aerial Imagery와 Annotation를 통합 및 정제하는 코드이다.
2024년 이전 Bass Connection팀이 생성한 코드(https://github.com/slaicha/cape_town_segmentation)를 토대로 데이터 Preparation pipeline을 개선하였음. 최종 산출물은 gpkg 파일이며, 268개 19K의 Raw Annotation을 확보. 해당 Annotation의 데이터 품질은 뒤이어 1. Annotation Checking에서 시행함.
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
