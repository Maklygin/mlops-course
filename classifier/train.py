import wget
import zipfile
import torch
from torchvision import transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
from collections import defaultdict
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score
import os
from trainer import CNNRunner

def download_data() -> None:
    url = 'https://www.dropbox.com/s/gqdo90vhli893e0/data.zip?dl=1'
    wget.download(url, out='dataset')

    with zipfile.ZipFile('dataset/data.zip', 'r') as zip_ref:
        zip_ref.extractall('dataset')

class Flatten(nn.Module):
    def forward(self, x):
        # finally we have it in pytorch
        return torch.flatten(x, start_dim=1)


if __name__ == '__main__':
    download_data()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Path to a directory with image dataset and subfolders for training, validation and final testing
    DATA_PATH = r"dataset"
    NUM_WORKERS = 4
    SIZE_H = SIZE_W = 96
    NUM_CLASSES = 2
    EPOCH_NUM = 30
    BATCH_SIZE = 256

    image_mean = [0.485, 0.456, 0.406]
    image_std  = [0.229, 0.224, 0.225]

    EMBEDDING_SIZE = 128

    transformer = transforms.Compose([
        transforms.Resize((SIZE_H, SIZE_W)),        # scaling images to fixed size
        transforms.ToTensor(),                      # converting to tensors
        transforms.Normalize(image_mean, image_std) # normalize image data per-channel
    ])

    # load dataset using torchvision.datasets.ImageFolder
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(DATA_PATH, 'train_11k'), transform=transformer)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(DATA_PATH, 'val'), transform=transformer)
    test_dataset = torchvision.datasets.ImageFolder(os.path.join(DATA_PATH, 'test_labeled'), transform=transformer)
    n_train, n_val, n_test = len(train_dataset), len(val_dataset), len(test_dataset)

    train_batch_gen = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=NUM_WORKERS)

    val_batch_gen = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=BATCH_SIZE,
                                                num_workers=NUM_WORKERS)

    model = nn.Sequential()
    model.add_module('flatten', Flatten())
    model.add_module('dense1', nn.Linear(3 * SIZE_H * SIZE_W, 256))
    model.add_module('dense1_relu', nn.ReLU())
    model.add_module('dropout1', nn.Dropout(0.1))
    model.add_module('dense3', nn.Linear(256, EMBEDDING_SIZE))
    model.add_module('dense3_relu', nn.ReLU())
    model.add_module('dropout3', nn.Dropout(0.1))
    model.add_module('dense4_logits', nn.Linear(EMBEDDING_SIZE, NUM_CLASSES))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    ckpt_name = 'models/model_base.ckpt'
    model = model.to(device)

    runner = CNNRunner(model, opt, device, ckpt_name)
    
    runner.train(train_batch_gen, val_batch_gen, n_epochs=3, visualize=False)

