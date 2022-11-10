from typing import Union

#import streamlit as st\
import easydict
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from fastapi import FastAPI

import model.model as module_arch
from parse_config import ConfigParser



def load_model():
    # device = 'cpu'
    # #device = torch.device("cpu")
    # PATH = 'checkpoint/model_best.pth'
    # model = EffNet().to(device)
    # model.load_state_dict(torch.load(PATH, map_location=device))
    device = 'cpu'
    # build model architecture
    model = config.init_obj('arch', module_arch)

    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    return model

def transform(image):
    cv2_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    image = Image.fromarray(cv2_img)

    return transform(image)


args = easydict.EasyDict({
        "config" : "checkpoint/config.json",
        "resume" : "checkpoint/model_best.pth"
    })

config = ConfigParser.from_args(args)
#main(config)

label_dict = {
        0: 'Heart',
        1: 'Oblong',
        2: 'Oval',
        3: 'Round',
        4: 'Square'
    }

model = load_model()

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict")
def predict(model, image):
    imgArray = np.array(image)
    input = transform(imgArray)
    input = input.unsqueeze(dim=0)
    output = model(image)
    output_idx = torch.argmax(output)

    return label_dict[output_idx.item()]



    # args = argparse.ArgumentParser(description='Face Shape - EfficientNet B7')
    # args.add_argument('-c', '--config', default=None, type=str,
    #                   help='config file path (default: None)')
    # args.add_argument('-r', '--resume', default=None, type=str,
    #                   help='path to latest checkpoint (default: None)')
    # args.add_argument('-d', '--device', default=None, type=str,
    #                   help='indices of GPUs to enable (default: all)')
    