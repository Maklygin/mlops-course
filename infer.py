import os

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from omegaconf import DictConfig
from torchvision import transforms

from classifier.trainer import CNNRunner


class Flatten(nn.Module):
    def forward(self, x):
        # finally we have it in pytorch
        return torch.flatten(x, start_dim=1)


@hydra.main(config_path="configs", config_name="conf.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:

    with open(cfg.model.ckpt_name, "rb") as f:
        best_model = torch.load(f)

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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    runner = CNNRunner(best_model, None, device, cfg.model.ckpt_name)
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(cfg.data.data_path, cfg.data.test_path), transform=transformer
    )

    test_batch_gen = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers
    )

    runner.validate(test_batch_gen, best_model, phase_name="test")
    scores = runner.get_scores()

    pd.DataFrame(data={"class": np.int32(scores > 0.5)}).to_csv(
        cfg.data.preds_path, index=False
    )


if __name__ == "__main__":
    main()
