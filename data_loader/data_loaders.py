import cv2
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
#from base import BaseDataLoader
# Dataset
class FaceShapeDataset(Dataset):
    def __init__(self, df, transform=None, split="train"):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.split = split
        class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        self.label2idx = {class_name:i for i, class_name in enumerate(class_names)}
        self.idx2label = {v:k for k,v in self.label2idx.items()}
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            img_path = self.df.loc[idx, "path"]
            #img = Image.open(img_path).convert('L')
            cv2_img = cv2.imread(img_path)
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            
            #cv2_img = cv2.Canny(cv2_img, 30, 180)
            img=Image.fromarray(cv2_img)
            if self.transform:
                img = self.transform(img)
            
            label = self.df.loc[idx, "label"]
            label = self.label2idx[label]
            return img, torch.tensor(label)
        except:
            print(f"Error load image {img_path}")
            idx = 0
            img_path = self.df.loc[idx, "path"]
            #img = Image.open(img_path).convert('L')
            cv2_img = cv2.imread(img_path)
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            
            #cv2_img = cv2.Canny(cv2_img, 30, 180)
            img=Image.fromarray(cv2_img)
            if self.transform:
                img = self.transform(img)
            
            label = self.df.loc[idx, "label"]
            label = self.label2idx[label]
            return img, torch.tensor(label)

'''
class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class FaceShapeDataLoader(DataLoader):
    """
    Face Shape data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size=128, shuffle=True):
        trsfm = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = FaceShapeDataset(self.data_dir, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle)
'''

def load_train_data(
            data_dir,
            batch_size=32,
            shuffle=True
            ):
    train_df = pd.read_csv(data_dir)

    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    return DataLoader(
                    FaceShapeDataset(train_df, transform=transform, split="train"),
                    batch_size=batch_size, shuffle=shuffle
                    )

def load_val_data(
            data_dir,
            batch_size=32,
            shuffle=False
            ):
    val_df = pd.read_csv(data_dir)

    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    return DataLoader(
                    FaceShapeDataset(val_df, transform=transform, split="val"),
                    batch_size=batch_size, shuffle=shuffle
                    )