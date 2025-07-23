import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from model_M import PVModel
import argparse
from data_gen_M import get_training_augmentation, get_validation_augmentation, Dataset
from tqdm import tqdm
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'finetune'], help='Mode: train or finetune')
args = parser.parse_args()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' 

DATA_DIR = "/home/cmn60/cape_town_segmentation/output5k_stratified"

x_train_dir = os.path.join(DATA_DIR, "train/images")
y_train_dir = os.path.join(DATA_DIR, "train/masks")

x_valid_dir = os.path.join(DATA_DIR, "val/images")
y_valid_dir = os.path.join(DATA_DIR, "val/masks")

x_test_dir = os.path.join(DATA_DIR, "test/images")
y_test_dir = os.path.join(DATA_DIR, "test/masks")

CLASSES = ['background', 'PV_normal', 'PV_heater', 'PV_pool']

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
)

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
)


# Adjust batch_size based on your GPU memory
BATCH_SIZE = 16  
NUM_WORKERS = 4  

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=True,  
    drop_last=True    
)

valid_loader = DataLoader(
    valid_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)


EPOCHS = 50
T_MAX = EPOCHS * len(train_loader)
OUT_CLASSES = len(train_dataset.CLASSES)  # Should be 4: [background, PV_normal, PV_heater, PV_pool]

if args.mode == "train":
    # Model initialization
    model = PVModel(
        arch="FPN", 
        encoder_name="resnext50_32x4d", 
        in_channels=3, 
        out_classes=OUT_CLASSES
    )
elif args.mode == "finetune":
    model = PVModel.load_from_checkpoint(
        checkpoint_path="/home/cmn60/cape_town_segmentation/logs/pv_segmentation/version_7/checkpoints/pv-model-epoch=37-valid_dataset_iou=0.9816.ckpt",
        arch='FPN',
        encoder_name='resnext50_32x4d',
        in_channels=3,
        out_classes=OUT_CLASSES
    )

print(f"Number of classes: {OUT_CLASSES}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batches per epoch: {len(train_loader)}")



callbacks = [
    # Save best model based on validation IoU
    ModelCheckpoint(
    monitor='valid_avg_PV_iou',
    mode='max',
    filename='pv-model-{epoch:02d}-{valid_avg_PV_iou:.4f}',
    save_last=True
    ),
    
    
    # Learning rate monitoring
    LearningRateMonitor(logging_interval='epoch')
]


logger = TensorBoardLogger(
    save_dir='./logs',
    name='pv_segmentation',
    version=None
)

# Enhanced trainer configuration
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    callbacks=callbacks,
    devices=1,
    logger=logger,
    log_every_n_steps=10,  
    check_val_every_n_epoch=1,
    accumulate_grad_batches=1,  
    precision='16-mixed',  
    gradient_clip_val=1.0,  
    deterministic=False,  
    enable_progress_bar=True,
    enable_model_summary=True,
)

print(f"Training setup:")
print(f"- Model: {model.model.__class__.__name__} with resnext50_32x4d encoder")
print(f"- Classes: {OUT_CLASSES} ({train_dataset.CLASSES})")
print(f"- Max epochs: {EPOCHS}")
print(f"- Batch size: {BATCH_SIZE}")
print(f"- Training samples: {len(train_dataset)}")
print(f"- Validation samples: {len(valid_dataset)}")
print(f"- Batches per epoch: {len(train_loader)}")

# Start training
print("\nStarting training...")
trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=valid_loader,
)


# Load the best model for testing
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"\nBest model saved at: {best_model_path}")

# Load best model
best_model = PVModel.load_from_checkpoint(
    best_model_path,
    arch="FPN",
    encoder_name="resnext50_32x4d",
    in_channels=3,
    out_classes=OUT_CLASSES,
)

# Test the BEST model (not the last epoch model)
print("\nTesting the BEST model...")
trainer.test(best_model, dataloaders=test_loader)

# Continue with inference using the best model
print("\nRunning inference with the best model...")

# JSON folder
OUTPUT_DIR = "./prediction_outputs_v33_5k"
# Color-coded segmentation mask folder
COLOR_MASK_DIR = "./predicted_masks_color_v33_5k"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COLOR_MASK_DIR, exist_ok=True)

# Define color mapping for predicted masks
colors = {
    0: (0, 0, 0),       # background - black
    1: (0, 255, 0),     # PV_normal - green
    2: (0, 0, 255),     # PV_heater - red (OpenCV uses BGR)
    3: (255, 0, 0),     # PV_pool - blue
}

def decode_mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in colors.items():
        color_mask[mask == class_idx] = color
    return color_mask

# Put model in eval mode
best_model.eval()
# JSON + Color PNG saving
with torch.no_grad():
    for i, (images, masks) in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = images.float().to(best_model.device)
        masks = masks.to(best_model.device)

        logits = best_model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        gt_masks = masks.cpu().numpy()

        for b in range(images.shape[0]):
            image_index = i * test_loader.batch_size + b
            pred_mask = preds[b]
            gt_mask = gt_masks[b]

            # Get original image name
            image_name = os.path.basename(test_dataset.images_fps[image_index])
            base_name = os.path.splitext(image_name)[0]

            per_class_coords = {
                "image_index": image_index,
                "image_name": image_name,
                "predicted_coords": {},
                "ground_truth_coords": {},
            }

            for cls in range(best_model.number_of_classes):
                pred_coords = np.column_stack(np.where(pred_mask == cls))
                gt_coords = np.column_stack(np.where(gt_mask == cls))
                per_class_coords["predicted_coords"][f"class_{cls}"] = pred_coords.tolist()
                per_class_coords["ground_truth_coords"][f"class_{cls}"] = gt_coords.tolist()

            # Save compact JSON with image name
            json_path = os.path.join(OUTPUT_DIR, f"{image_name}.json")
            with open(json_path, "w") as f:
                json.dump(per_class_coords, f, separators=(",", ":"))

            # Save color-coded predicted mask
            color_mask = decode_mask_to_color(pred_mask)
            save_path = os.path.join(COLOR_MASK_DIR, f"{base_name}_pred.png")
            cv2.imwrite(save_path, color_mask)

print("✅ Prediction masks and JSON files saved.")
