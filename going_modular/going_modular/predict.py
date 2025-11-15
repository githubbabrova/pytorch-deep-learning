
import torch

from torchvision import transforms
from utils import load_model

import argparse

parser = argparse.ArgumentParser(description="Predict a PyTorch image class.")

parser.add_argument("--image_path", type=str, default="data/pizza_steak_sushi/test/pizza/1000.jpg", help="Path to the image to predict")
parser.add_argument("--model_path", type=str, default="models/05_going_modular_script_mode_tinyvgg_model.pth", help="Path to the trained model")
args = parser.parse_args()


model = load_model(model=torch.nn.Module(),
                   model_path=args.model_path,
                   device=torch.device("cpu"))
model.eval()    