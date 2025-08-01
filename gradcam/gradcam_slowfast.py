import os
import cv2
import torch
import numpy as np
from pathlib import Path
# 1) Import setup routines
from slowfast_setup import load_dataset, load_model, evaluate_single_sample
from slowfast_setup import 