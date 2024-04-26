import cv2
import numpy as np
import torch

from backbones import get_model


class Inference:
    def __init__(self, weight='models/backbone.pth'):
        self.weight = weight
        self.net = get_model(fp16=False)
        self.net.load_state_dict(torch.load(self.weight,map_location='cpu'))
        self.net.eval()

    @torch.no_grad()
    def inference(self, img: np.ndarray):
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        feat = self.net(img).numpy()
        return feat


