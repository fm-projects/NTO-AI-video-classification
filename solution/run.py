import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import clip
from clip.model import build_model

import platform
import os
import sys
import argparse
from collections import Counter

import warnings
warnings.simplefilter("ignore")


VIDEO_CLASSES = [
    "water",
    "car",
    "cloud",
    "food",
    "flower",
    "dance",
    "animal",
    "sunset",
    "fire",
]


CONFIG = {
    "FRAMES": 8,
    "FRAMES_REMOVE_FIRST_LAST": True,
    "SIZE": (224, 224),
    "WEIGHT_PATH": "./weights.pt",
    "TEST_PATH": "./input/input.csv",
    "VIDEO_PATH": "/video",
    "OUTPUT_PATH": "./output/",
    "OUTPUT_NAME": "predictions.csv"
}


def get_preprocess():
    """
    CLIP"s default image preprocessing function

    Source:
    https://github.com/openai/CLIP/blob/main/clip/clip.py
    """
    return Compose([
        Resize(size=CONFIG["SIZE"][0], interpolation=InterpolationMode.BICUBIC,
               max_size=None, antialias=None),
        CenterCrop(size=CONFIG["SIZE"]),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                  std=(0.26862954, 0.26130258, 0.27577711))
    ])


def read_video(path):
    """
    Read a video

    Arguments:
        path: Path of a video

    Returns:
        Frames of a video
    """
    frames = []
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    N = length // CONFIG["FRAMES"]
    current_frame = 1
    for i in range(length):
        ret, frame = cap.read(current_frame)
        if ret and i == current_frame and len(frames) < CONFIG["FRAMES"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, CONFIG["SIZE"])
            frames.append(frame)
            current_frame += N
    cap.release()

    if CONFIG.get("FRAMES_REMOVE_FIRST_LAST"):
        return frames[1:-1]
    return frames


def predict(model, frames, preprocess, text, device):
    """
    Make a prediction.

    Arguments:
        model: CLIP model
        frames: Frames of a video
        preprocess: Function for preprocessing/transforming frames
        text: CLIP text tokens
        device: Available device (GPU/CPU)
    """
    ans = []

    for frame in frames:
        image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)

        with torch.no_grad():
            model.encode_image(image)
            model.encode_text(text)
            logits_per_image, _ = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        ans.append(np.where(probs[0] == probs[0].max())[0][0])

    # Select the most common video class in prediction as the answer
    return Counter(ans).most_common()[0][0]


def main(args, device):
    # Load dataset with video filenames
    df_test = pd.read_csv(args.test_path, index_col=0)

    # Load the model
    state = torch.load(args.weight_path, map_location="cpu")

    # Build the model
    model = build_model(state).to(device)

    # Create CLIP text tokens
    text = clip.tokenize(VIDEO_CLASSES).to(device)

    # Image preprocessing function
    preprocess = get_preprocess()

    pred = []
    for path in df_test.path:
        video = read_video(f'{args.video_path}/{path}')
        pred.append(predict(model, video, preprocess, text, device))

    # Save prediction
    df_test["labels"] = pred
    os.makedirs(args.output_path, exist_ok=True)
    df_test["labels"].to_csv(args.output_path + CONFIG["OUTPUT_NAME"])


def create_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_path", default=CONFIG["TEST_PATH"], type=str)
    parser.add_argument(
        "--weight_path", default=CONFIG["WEIGHT_PATH"], type=str)
    parser.add_argument(
        "--output_path", default=CONFIG["OUTPUT_PATH"], type=str)
    parser.add_argument(
        "--video_path", default=CONFIG["VIDEO_PATH"], type=str)
    return parser


if __name__ == "__main__":
    args = create_args_parser().parse_args()

    if torch.cuda.is_available():
        print("Using GPU: {}\n".format(torch.cuda.get_device_name()))
        device = torch.device("cuda")
    else:
        print("\nGPU not found. Using CPU: {}\n".format(platform.processor()))
        device = torch.device("cpu")

    main(args, device)
    sys.exit(0)
