# U-Net Multiclass Segmentation

## Step 1: Data Generation
Before running U-Net, you need to create image and mask pairs. Run mask_gen.ipynb to create a dataset folder like this: 
```
dataset/
├── test/
│   ├── images/
│   └── masks/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```
The masks are one-hot encoded. They look like black squares. They have pixels labeled 0 for background, 1 for PV_normal, 2 for PV_heater, 3 for PV_pool.

Run visualize.ipynb to see the masks in color.

## Step 2: Train the U-Net (multiclass)
The files needed to train the U-Net are train_M.py, model_M.py, and data_gen_M.py, as well as the dataset folder from step 1\
Change the file path for OUTPUT_DIR, COLOR_MASK_DIR\
In terminal, type: python3 train_M.py --mode train\
If you want to resume training, update checkpoint_path and type “python3 train_M.py --mode finetune”\
This will produce:
* metrics_log_{timestamp}.csv: Logs metrics (IoU, precision, recall, etc.) during training
* prediction_outputs: A folder of jsons where each json has predicted coordinates (pixels) and ground truth coordinates (pixels)
* predicted_masks_color: A folder of prediction masks that are color-coded
* logs: A folder with a ckpt for the best model and ckpt for last model (best model is saved based on valid_avg_PV_iou which is the average of the IoUs of PV_normal, PV_heater, and PV_pool)

## Step 3: Make predictions on an unseen dataset using a trained U-Net model
### Option a. Make a prediction with the ground truth
The files needed are predict.py and data_gen_M.py\
In main(), change the file paths for --ckpt, --images, **--masks**, --outdir, --maskdir, and **--csv**\
In terminal, type: python3 predict.py

### Option b. Make a prediction without the ground truth
The files needed are pred3.py and inference_dataset.py (inference_dataset.py has a dataset class that does not require ground truth masks)\
In main(), change the file paths for --ckpt, --images, --outdir, --maskdir\
In terminal, type: python3 pred3.py
