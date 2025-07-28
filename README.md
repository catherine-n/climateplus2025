# Energy Transition During Energy Crisis: Cape Town's Experience

**Overview**

This project aims to develop a **computer vision prototype** to detect **rooftop solar energy systems** in Cape Town, South Africa.
Due to ongoing electricity instability, many households have installed private solar systems to mitigate the impact of load-shedding (scheduled power outages).
This model estimates the distribution of rooftop solar installations at the neighborhood levels. This could provide insights to support empirical research on energy inequality in Cape Town.

**Modeling**

Two distinct deep learning pipelines were developed to identify and map rooftop systems such as solar panels, water heaters, and pool heaters. 
* The **first pipeline(blue)** combines **YOLOx_11_OBB**(Oriented Bounding Box) for **object detection** with **U-Net** for **pixel-wise segmentation**. YOLOx_11_OBB localizes each rooftop system, and U-Net refines these detections by classifying pixels.

* The **second pipeline(orange)** is a **standalone semantic segmentation** model based solely on **U-Net**, utilizing ReNext50 and a Feature Pyramid Network (FPN). It directly segments solar panel area in aerial images through pixel-wise classification.
<img width="1698" height="406" alt="Image" src="https://github.com/user-attachments/assets/3a279bb8-ea07-45d1-8816-b127175cb848" />

In both pipelines, the segmentation results are post-processed with geospatial data to map rooftop solar panels for subsequent econometric analysis.

**Test result**

The evaluation was conduced in three districts of Cape Town (1 km² per district, ground truth: 337 solar panels). <br>
In terms of solar panel area capture accuracy :
* The YOLO + U-Net : 71.2% accuracy.
* The U-Net-only : 99.7% accuracy.

**Conclusion**

While the development of a streamlined data pipeline is promising, the test result is based on **area-level accuracy** over 1km² grids rather than object-level performance. To support large-scale empirical research, the following limitations must be addressed:

* Developing object-level evaluation metrics
* Enhancing detection and segmentation accuracy at the object level
* Improving model robustness across diverse rooftop conditions

_For more informations, see the full project poster: [36x42_Energy_Transition_July25.pdf](https://github.com/user-attachments/files/21447520/36x42_Energy_Transition_July25.pdf)_

---------------------

**File structure**
```
├─ 0.Data_processing
├─ 1.Annotation_checking
├─ 2.Modeling
│  ├─ 2.1.YOLO
│  │  ├─ Multi_Class
│  │  └─ Single_Class
│  ├─ 2.2.U-Net
│  └─ 2.3.YOLO+U-Net
├─ 3.Post_Processing
├─ 4.Results
└─ README.md
```

**0. Data_processing**: Integrates and refines aerial imagery and annotation data.

**1. Annotation_checking**: Identifies misclassified annotations and applies necessary updates.

**2. Modeling**  
- **2.1 YOLO (Object Detection)**  
  • Multi-class detection *(not applied in this pipeline)*: Solar Panel, Water Heater, Pool Heater  
  • Single-class detection: All solar energy systems as one class  
- **2.2 U-Net (Semantic Segmentation)**  
  • Performs multi-class segmentation  
- **2.3 YOLO + U-Net (Objecte Detection → Semantic Segmentation)**  
  • Crops detected objects from YOLO output for U-Net segmentation

**3. Post_Processing**: Converts pixel coordinates to GPS coordinates and calculates surface area.

**4. Results**: Project poster
