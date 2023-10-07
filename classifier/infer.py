import torch
from trainer import CNNRunner
import torchvision
import os
import torch.nn as nn
from torchvision import transforms
import numpy as np
import pandas as pd


class Flatten(nn.Module):
    def forward(self, x):
        # finally we have it in pytorch
        return torch.flatten(x, start_dim=1)


if __name__ == "__main__":
    ckpt_name = "models/model_base.ckpt"
    with open(ckpt_name, "rb") as f:
        best_model = torch.load(f)

    DATA_PATH = r"dataset"
    NUM_WORKERS = 4
    SIZE_H = SIZE_W = 96
    NUM_CLASSES = 2
    EPOCH_NUM = 30
    BATCH_SIZE = 256

    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    EMBEDDING_SIZE = 128

    transformer = transforms.Compose(
        [
            transforms.Resize((SIZE_H, SIZE_W)),  # scaling images to fixed size
            transforms.ToTensor(),  # converting to tensors
            transforms.Normalize(
                image_mean, image_std
            ),  # normalize image data per-channel
        ]
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    runner = CNNRunner(best_model, None, device, ckpt_name)
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(DATA_PATH, "test_labeled"), transform=transformer
    )

    test_batch_gen = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    test_stats = runner.validate(test_batch_gen, best_model, phase_name="test")
    scores = runner.get_scores()

    pd.DataFrame(data={"class": np.int32(scores > 0.5)}).to_csv(
        "dataset/preds.csv", index=False
    )
