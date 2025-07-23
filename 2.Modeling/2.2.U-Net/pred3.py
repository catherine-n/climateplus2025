import os, argparse, json, cv2, torch, numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from model_M import PVModel
from data_gen_pred import get_validation_augmentation
from inference_dataset import InferenceDataset # Dataset class without masks

# This file is for predictions with NO ground truth (NO one-hot encoded masks)

COLORS = { # BGR (OpenCV) format
    0: (0, 0, 0),        # background - black
    1: (0, 255, 0),      # PV_normal - green
    2: (0, 0, 255),      # PV_heater - red
    3: (255, 0, 0),      # PV_pool - blue
}

def decode_mask_to_color(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, bgr in COLORS.items():
        color[mask == idx] = bgr
    return color

def main():
    parser = argparse.ArgumentParser()
    # Checkpoint of a trained model
    parser.add_argument("--ckpt", default="/home/cmn60/cape_town_segmentation/logs/pv_segmentation/version_28/checkpoints/pv-model-epoch=38-valid_avg_PV_iou=0.9572.ckpt", help="Path to last.ckpt")
    
    # Input images
    parser.add_argument(
        "--images",
        default="/shared/data/climateplus2025/Prediction_for_poster_July21/Cropped_Images_320_centered_from_YOLO_for_unet_prediction",
        help="Folder with RGB tiles to predict on",
    )
    parser.add_argument("--batch_size", type=int, default=8)

    # JSON predictions output directory
    parser.add_argument("--outdir", default="prediction_outputs_v48")
    
    # Mask predictions output directory (color-coded)
    parser.add_argument("--maskdir", default="prediction_masks_v48")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.maskdir, exist_ok=True)

    CLASSES = ["background", "PV_normal", "PV_heater", "PV_pool"]
    n_classes = len(CLASSES)

    dataset = InferenceDataset(
        image_dir=args.images,
        augmentation=get_validation_augmentation()
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PVModel.load_from_checkpoint(
        args.ckpt,
        arch="FPN",
        encoder_name="resnext50_32x4d",
        in_channels=3,
        out_classes=n_classes,
    ).to(device)
    model.eval()

    with torch.no_grad():
        for i, imgs in tqdm(enumerate(loader), total=len(loader)):
            imgs = imgs.float().to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            for b in range(preds.size(0)):
                idx = i * loader.batch_size + b
                img_path = dataset.image_paths[idx]
                base = os.path.splitext(os.path.basename(img_path))[0]
                pred_np = preds[b].cpu().numpy()

                color = decode_mask_to_color(pred_np)
                cv2.imwrite(os.path.join(args.maskdir, f"{base}_pred.png"), color)

                coords = {
                    "image_index": idx,
                    "image_name": os.path.basename(img_path),
                    "predicted_coords": {
                        CLASSES[cls]: np.column_stack(np.where(pred_np == cls)).tolist()
                        for cls in range(n_classes)
                    }
                }
                with open(os.path.join(args.outdir, f"{base}.json"), "w") as f:
                    json.dump(coords, f, separators=(",", ":"))

    print(f"✅ Predictions saved to {args.maskdir}")
    print(f"✅ Coordinate data saved to {args.outdir}")
    print("🎉 All done – predictions are ready!")

if __name__ == "__main__":
    main()
