# climateplus2025
--------------------
**Change log**\
(07.22) Created folder structure as below. Please git pull first to import. You can create subfolders within the structure as needed.\
(07.22) Make sure to create your own branch first, and then submit a pull requests to merge it into `main` branch


---------------------
Folder Structure (Draft)


**0.Data_processing (Original 19K dataset)**
  - raw, interim, final .... (docs, code)

**1.Annotation_checking**
  - Annotation_checker (docs, code)
  - Confusion_matrix (docs, code)
    
**2.Modeling**\ 
**2.1 YOLO** (data;test/train/val, model, artifacts, evaluation, visualization, docs)\
**2.2 U-Net** (data;test/train/val, model, artifacts, evaluation, visualization, docs)\
**2.3 YOLO + U-Net** (model, prediction, evaluation, visualization, docs)

**3.Post_processing**
  - model, output (docs, code)

**4.Results**
  - Organized output (models' metrics, predicted solar panel area)
  - Poster and others
