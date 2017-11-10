from PIL import Image
import numpy as np
import sys

def load_training_data(original_path, mask_path):
    original = Image.open(original)
    mask = Image.open(mask)
