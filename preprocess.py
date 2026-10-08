"""The one way a cashew crop is prepared for the CNN.

Used by live inference (cashew_cnn_grader.py), the auto-labeler and training, so the model
always sees the same framing: tight box around the cashew, padded to a square, 224 x 224,
ImageNet normalisation.
"""
import cv2
import numpy as np

CNN_INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CASHEW_GRAY_THRESHOLD = 70      # bright cashew against the dark roller
MIN_CASHEW_AREA_PX = 150
TIGHT_MARGIN_PX = 4


def find_cashew_rect(crop_bgr):
    """(area, (x, y, w, h)) of the largest bright blob in the crop, or (0, None)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0, None
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, CASHEW_GRAY_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area <= 100:
        return 0, None
    return int(area), cv2.boundingRect(c)


def tighten(crop_bgr, margin=TIGHT_MARGIN_PX):
    """Crop to the cashew itself plus a small margin. Returns the input when no cashew is found."""
    area, rect = find_cashew_rect(crop_bgr)
    if rect is None or area < MIN_CASHEW_AREA_PX:
        return crop_bgr
    x, y, w, h = rect
    H, W = crop_bgr.shape[:2]
    return crop_bgr[max(0, y - margin):min(H, y + h + margin), max(0, x - margin):min(W, x + w + margin)]


def to_square(crop_bgr):
    """Pad the shorter side (edge pixels repeated) so the cashew keeps its shape when resized."""
    h, w = crop_bgr.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(crop_bgr, top, side - h - top, left, side - w - left, cv2.BORDER_REPLICATE)


def prepare_crop(crop_bgr, tight=True):
    """BGR crop -> RGB uint8 array, square, ready for the transform. `tight=False` when the
    caller has already cut the crop to the cashew (live inference does)."""
    if tight:
        crop_bgr = tighten(crop_bgr)
    return cv2.cvtColor(to_square(crop_bgr), cv2.COLOR_BGR2RGB)


def build_transform(train=False):
    """torchvision transform applied after prepare_crop(). Augmentation only when train=True."""
    from torchvision import transforms
    steps = [transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE))]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
                  transforms.ColorJitter(brightness=0.1, contrast=0.1)]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def crop_to_tensor(crop_bgr, transform, tight=True):
    from PIL import Image
    return transform(Image.fromarray(prepare_crop(crop_bgr, tight=tight)))


def load_cnn_checkpoint(path, device="cpu"):
    """(model in eval mode, class_names) from a MobileNetV2 checkpoint of this project."""
    import torch
    from torchvision import models
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    class_names = [str(n) for n in ckpt["class_names"]]
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
    state = ckpt.get("model_state_dict")
    if state is None:
        state = ckpt["model_state"]
    model.load_state_dict(state)
    return model.eval().to(device), class_names
