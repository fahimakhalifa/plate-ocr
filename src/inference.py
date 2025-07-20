import torch
from torch import Tensor
from torchvision import transforms
from PIL import Image
import json
from model import PlateRecognitionModel
from dataset import decode

def load_crnn_model():
    with open("model/char_to_idx.json") as f:
        char_to_idx = json.load(f)
    idx_to_char = {int(v): k for k, v in char_to_idx.items()}

    model = PlateRecognitionModel(vocab_size=len(char_to_idx))
    model.load_state_dict(torch.load("model/plate_model_v1.pth", map_location="cpu"))
    model.eval()
    return model, idx_to_char, char_to_idx

transform = transforms.Compose([
    transforms.Resize((32, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

with open("model/char_to_idx.json") as f:
        char_to_idx = json.load(f)
idx_to_char = {int(v): k for k, v in char_to_idx.items()}

model = PlateRecognitionModel(vocab_size=len(char_to_idx))
model.load_state_dict(torch.load("plate_model_v1.pth", map_location="cpu"))
model.eval()

def predict_plate(tensor_image):
    with torch.no_grad():
        logits = model(tensor_image)  # [1, T, C]
        pred = logits.argmax(2).squeeze(1)  # [T]
    return decode(pred, idx_to_char)

if __name__ == "__main__":
    path = "cropped_plate_0.png"  
    print("Prediction:", predict_plate(path))
