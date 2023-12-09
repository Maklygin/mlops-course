import os
import zipfile

import wget
from dvc.api import DVCFileSystem


def download_data() -> None:
    url = "https://www.dropbox.com/s/gqdo90vhli893e0/data.zip?dl=1"
    wget.download(url, out="data")

    with zipfile.ZipFile("data/data.zip", "r") as zip_ref:
        zip_ref.extractall("data")


def get_dataset_from_dvc_and_unpuck() -> None:
    fs = DVCFileSystem(".dvc")
    fs.get_file("data/data.zip", "data/data.zip")

    with zipfile.ZipFile("data/data.zip", "r") as zip_ref:
        zip_ref.extractall("data")

    os.remove("data/data.zip")
