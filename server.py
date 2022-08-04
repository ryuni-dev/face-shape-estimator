import argparse
from zmq import device
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import model.model as module_arch
from parse_config import ConfigParser


def load_model(config):
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

def predict(model, image):
    output = model(image)
    return output

def main(config):
    model = load_model(config)
    #model = model.to(device)
    model.eval()

    label_dict = {
            0: 'Heart',
            1: 'Oblong',
            2: 'Oval',
            3: 'Round',
            4: 'Square'
        }

    st.write("""
        # Face Shape Estimator
    """)
    uploaded_file = st.file_uploader("Upload face Image!!")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()

        col1, col2= st.columns(2)     

        image = Image.open(uploaded_file)
        with col1:
            st.header("Input face image")
            st.image(image)

        with col2:
            st.header("Result")
            imgArray = np.array(image)
            input = transform(imgArray)
            input = input.unsqueeze(dim=0)
            
            res = predict(model, input)

            res_idx = torch.argmax(res)
            st.write(f'Your Face Shape is "{label_dict[res_idx.item()]}"')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Face Shape - EfficientNet B7')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)