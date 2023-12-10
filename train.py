import os

import hydra
import torch
import torch.nn as nn
import torchvision
from omegaconf import DictConfig
from torchvision import transforms

from classifier.trainer import CNNRunner
from dataset.utils import get_dataset_from_dvc_and_unpuck


class Flatten(nn.Module):
    def forward(self, x):
        # finally we have it in pytorch
        return torch.flatten(x, start_dim=1)


@hydra.main(config_path="configs", config_name="conf.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:

    get_dataset_from_dvc_and_unpuck()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    transformer = transforms.Compose(
        [
            transforms.Resize(
                (cfg.data.size_h, cfg.data.size_w)
            ),  # scaling images to fixed size
            transforms.ToTensor(),  # converting to tensors
            transforms.Normalize(
                image_mean, image_std
            ),  # normalize image data per-channel
        ]
    )

    # load dataset using torchvision.datasets.ImageFolder
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(cfg.data.data_path, cfg.data.train_path), transform=transformer
    )
    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(cfg.data.data_path, cfg.data.eval_path), transform=transformer
    )

    train_batch_gen = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )

    val_batch_gen = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers
    )

    model = nn.Sequential()
    model.add_module("flatten", Flatten())
    model.add_module("dense1", nn.Linear(3 * cfg.data.size_h * cfg.data.size_w, 256))
    model.add_module("dense1_relu", nn.ReLU())
    model.add_module("dropout1", nn.Dropout(0.1))
    model.add_module("dense3", nn.Linear(256, cfg.model.embedding_size))
    model.add_module("dense3_relu", nn.ReLU())
    model.add_module("dropout3", nn.Dropout(0.1))
    model.add_module(
        "dense4_logits", nn.Linear(cfg.model.embedding_size, cfg.data.num_classes)
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    opt.zero_grad()
    model = model.to(device)

    runner = CNNRunner(model, opt, device, cfg.model.ckpt_name)

    runner.train(
        train_batch_gen, val_batch_gen, n_epochs=cfg.train.epoch_num, visualize=False
    )


if __name__ == "__main__":
    main()
