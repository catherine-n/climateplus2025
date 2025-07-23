import os, argparse, json, cv2, torch, numpy as np, pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from model_M import PVModel
from data_gen_M import Dataset, get_validation_augmentation   # Dataset class with masks

# This file is for predictions with ground truth (one-hot encoded masks)

COLORS = {                 # BGR (OpenCV) format
    0: (  0,   0,   0),    # background – black
    1: (  0, 255,   0),    # PV_normal – green
    2: (  0,   0, 255),    # PV_heater – red
    3: (255,   0,   0),    # PV_pool   – blue
}

def decode_mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert 2‑D label mask to 3‑channel color image."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, bgr in COLORS.items():
        color[mask == idx] = bgr
    return color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/home/cmn60/cape_town_segmentation/logs/pv_segmentation/version_28/checkpoints/pv-model-epoch=38-valid_avg_PV_iou=0.9572.ckpt", help="Path to last.ckpt")
    
    # Input images
    parser.add_argument(
        "--images",
        default="/shared/data/climateplus2025/Prediction_for_poster_July21/Cropped_Images_320_centered_from_YOLO_for_unet_prediction",
        help="Folder with RGB tiles",
    )

    # Ground truth masks corresponding to input images
    parser.add_argument(
        "--masks",
        default="/shared/data/climateplus2025/Prediction_for_poster_July21/Prediction/masked_images",
        help="Folder with ground‑truth masks (same names as images)",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    
    # JSON predictions output directory
    parser.add_argument("--outdir", default="prediction_outputs_v46", help="JSON root")

    # Mask predictions output directory (color-coded)
    parser.add_argument("--maskdir", default="predicted_masks_color_v46", help="PNG root")
    parser.add_argument("--csv", default="metrics_pred_v46.csv")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.maskdir, exist_ok=True)

    
    CLASSES = ["background", "PV_normal", "PV_heater", "PV_pool"]
    n_classes = len(CLASSES)

    dataset = Dataset(
        args.images,
        args.masks,
        augmentation=get_validation_augmentation(),  # resize / normalise only
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PVModel.load_from_checkpoint(
        args.ckpt,
        arch="FPN",
        encoder_name="resnext50_32x4d",
        in_channels=3,
        out_classes=n_classes,
    ).to(device)
    model.eval()

    inter = torch.zeros(n_classes, dtype=torch.float64, device=device)
    union = torch.zeros_like(inter)
    tp = torch.zeros_like(inter)
    fp = torch.zeros_like(inter)
    fn = torch.zeros_like(inter)

    with torch.no_grad():
        for i, (imgs, gts) in tqdm(enumerate(loader), total=len(loader)):
            imgs = imgs.float().to(device)
            gts  = gts.to(device)

            logits = model(imgs)
            preds  = torch.argmax(logits, dim=1) 


            for cls in range(n_classes):
                pred_cls = preds == cls
                gt_cls   = gts  == cls
                inter[cls] += (pred_cls & gt_cls).sum()
                union[cls] += (pred_cls | gt_cls).sum()
                tp[cls]    += (pred_cls & gt_cls).sum()
                fp[cls]    += (pred_cls & ~gt_cls).sum()
                fn[cls]    += (~pred_cls & gt_cls).sum()


            for b in range(preds.size(0)):
                idx = i * loader.batch_size + b
                img_path = dataset.images_fps[idx]
                img_name = os.path.basename(img_path)
                base     = os.path.splitext(img_name)[0]
                pred_np  = preds[b].cpu().numpy()
                gt_np    = gts[b].cpu().numpy()

                # Color mask
                color = decode_mask_to_color(pred_np)
                cv2.imwrite(os.path.join(args.maskdir, f"{base}_pred.png"), color)

                # JSON with coordinates
                coords = {
                    "image_index": idx,
                    "image_name": img_name,
                    "predicted_coords": {},
                    "ground_truth_coords": {},
                }
                for cls in range(n_classes):
                    coords["predicted_coords"][CLASSES[cls]] = np.column_stack(
                        np.where(pred_np == cls)
                    ).tolist()
                    coords["ground_truth_coords"][CLASSES[cls]] = np.column_stack(
                        np.where(gt_np == cls)
                    ).tolist()
                with open(os.path.join(args.outdir, f"{base}.json"), "w") as f:
                    json.dump(coords, f, separators=(",", ":"))

    
    # Final metrics -> CSV
    eps = 1e-7
    iou       = (inter + eps) / (union + eps)
    precision = (tp    + eps) / (tp + fp + eps)
    recall    = (tp    + eps) / (tp + fn + eps)

    df = pd.DataFrame(
        {
            "class": CLASSES,
            "iou": iou.cpu().numpy(),
            "precision": precision.cpu().numpy(),
            "recall": recall.cpu().numpy(),
        }
    )
    df.to_csv(args.csv, index=False)
    print(f"✅ Saved class‑wise IoU / precision / recall to →  {args.csv}")

    print("🎉 All done – masks, JSONs and metrics are ready!")

# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
