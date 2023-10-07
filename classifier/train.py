import wget
import zipfile
import torch
from torchvision import transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
import numpy as np
import time
from collections import defaultdict
from tqdm.auto import tqdm

import os

def download_data() -> None:
    url = 'https://www.dropbox.com/s/gqdo90vhli893e0/data.zip?dl=1'
    wget.download(url, out='dataset')

    with zipfile.ZipFile('dataset/data.zip', 'r') as zip_ref:
        zip_ref.extractall('dataset')

class Runner():
    """Runner for experiments with supervised model."""
    def __init__(self, model, opt, device, checkpoint_name=None):
        self.model = model
        self.opt = opt
        self.device = device
        self.checkpoint_name = checkpoint_name
        
        self.epoch = 0
        self.output = None
        self.metrics = None
        self._global_step = 0
        self._set_events()
        self._top_val_accuracy = -1
        self.log_dict = {
            "train": [],
            "val": [],
            "test": []
        }
    
    def _set_events(self):
        """
        Additional method to initialize variables, which may store logging and evaluation info.
        The implementation below is extremely simple and only provided to help monitor performance.
        """
        self._phase_name = ''
        self.events = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list)
        }
    
    def _reset_events(self, event_name):
        self.events[event_name] = defaultdict(list)
    
    def forward(self, img_batch, **kwargs):
        """
        Forward method for your Runner.
        Should not be called directly outside your Runner.
        In simple case, this method should only implement your model forward pass.
        It should also return the model predictions and/or other meta info.
        
        Args:
            batch (mapping[str, Any]): dictionary with data batches from DataLoader.
            **kwargs: additional parameters to pass to the model.
        """
        logits = self.model(img_batch)
        output = {
            "logits": logits,
        }
        return output
    
    def run_criterion(self, batch):
        """
        Applies the criterion to the data batch and the model output, saved in self.output.
        
        Args:
            batch (mapping[str, Any]): dictionary with data batches from DataLoader.
        """
        raise NotImplementedError("To be implemented")
    
    def output_log(self):
        """
        Output log using the statistics collected in self.events[self._phase_name].
        Implement this method for logging purposes.
        """
        raise NotImplementedError("To be implemented")
    
    def _run_batch(self, batch):
        """
        Runs batch of data through the model, performing forward pass.
        This implementation performs data passing to necessary device and is adapted to the default pyTorch DataLoader.
        
        Args:
            batch (mapping[str, Any]): dictionary with data batches from DataLoader.
        """
        # split batch tuple into data batch and label batch
        X_batch, y_batch = batch

        # update the global step in iterations over source data
        self._global_step += len(y_batch)
        
        # move data to target device
        X_batch = X_batch.to(self.device)
        
        # run the batch through the model
        self.output = self.forward(X_batch)
    
    def _run_epoch(self, loader, train_phase=True, output_log=False, **kwargs):
        """
        Method that runs one epoch of the training process.
        
        Args:
            loader (DataLoader): data loader to iterate
            train_phase (bool): boolean value to determine if this is the training phase.
                Changes behavior for dropout, batch normalization, etc.
        """
        # Train phase
        # enable or disable dropout / batch_norm training behavior
        self.model.train(train_phase)
        
        _phase_description = 'Training' if train_phase else 'Evaluation'
        for batch in tqdm(loader, desc=_phase_description, leave=False):
            
            # forward pass through the model using preset device
            self._run_batch(batch)
            
            # train on batch: compute loss and gradients
            with torch.set_grad_enabled(train_phase):
                loss = self.run_criterion(batch)
            
            # compute backward pass if training phase
            # reminder: don't forget the optimizer step and zeroing the grads
            if train_phase:
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

        
        self.log_dict[self._phase_name].append(np.mean(self.events[self._phase_name]['loss']))
        
        if output_log:
            self.output_log(**kwargs)


from sklearn.metrics import f1_score, accuracy_score


class CNNRunner(Runner):
    def run_criterion(self, batch):
        """
        Applies the criterion to the data batch and the model output, saved in self.output.
        
        Args:
            batch (mapping[str, Any]): dictionary with data batches from DataLoader.
        """
        X_batch, label_batch = batch
        label_batch = label_batch.to(device)
        
        logit_batch = self.output['logits']
        
        # compute loss funciton
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logit_batch, label_batch)
        
        scores = F.softmax(logit_batch, 1).detach().cpu().numpy()[:, 1].tolist()
        labels = label_batch.detach().cpu().numpy().ravel().tolist()
        
        # log some info
        self.events[self._phase_name]['loss'].append(loss.detach().cpu().numpy())
        self.events[self._phase_name]['scores'].extend(scores)
        self.events[self._phase_name]['labels'].extend(labels)
        
        return loss
    
    def save_checkpoint(self):
        val_accuracy = self.metrics['accuracy']
        # save checkpoint of the best model to disk
        if val_accuracy > self._top_val_accuracy and self.checkpoint_name is not None:
            self._top_val_accuracy = val_accuracy
            torch.save(model, open(self.checkpoint_name, 'wb'))
    
    def output_log(self, **kwargs):
        """
        Output log using the statistics collected in self.events[self._phase_name].
        Let's have a fancy code for classification metrics calculation.
        """
        scores = np.array(self.events[self._phase_name]['scores'])
        labels = np.array(self.events[self._phase_name]['labels'])
        
        assert len(labels) > 0, print('Label list is empty')
        assert len(scores) > 0, print('Score list is empty')
        assert len(labels) == len(scores), print('Label and score lists are of different size')
        
        # visualize = kwargs.get('visualize', False)
        # if visualize:
        #     clear_output()
        
        self.metrics = {
            "loss": np.mean(self.events[self._phase_name]['loss']),
            "accuracy": accuracy_score(labels, np.int32(scores > 0.5)),
            "f1": f1_score(labels, np.int32(scores > 0.5))
        }
        print(f'{self._phase_name}: ', end='')
        print(' | '.join([f'{k}: {v:.4f}' for k, v in self.metrics.items()]))
        
        self.save_checkpoint() 
        

class Flatten(nn.Module):
    def forward(self, x):
        # finally we have it in pytorch
        return torch.flatten(x, start_dim=1)

    
    def train(self, train_loader, val_loader, n_epochs, model=None, opt=None, **kwargs):
        """
        Training process method, that runs for n_epochs over train_loader and performs validation using val_loader.
        
        Args:
            train_loader (DataLoader): training set data loader to iterate over
            val_loader (DataLoader): validation set data loader to iterate over
            n_epochs (int): epoch number to train for
            model (Model): torch nn.Module or nested class, that implements the model. Overwrites self.model.
            opt (Optimizer): torch optimizer to be used for loss minimization. Overwrites self.opt.
            **kwargs: additional parameters to pass to self.validate.
        """
        self.opt = (opt or self.opt)
        self.model = (model or self.model)
        
        for _epoch in range(n_epochs):
            start_time = time.time()
            self.epoch += 1
            print(f"epoch {self.epoch:3d}/{n_epochs:3d} started")
            
            # training part
            self._set_events()
            self._phase_name = 'train'
            self._run_epoch(train_loader, train_phase=True)
            
            print(f"epoch {self.epoch:3d}/{n_epochs:3d} took {time.time() - start_time:.2f}s")
            
            # validation part
            self._phase_name = 'val'
            self.validate(val_loader, **kwargs)
            self.save_checkpoint()
    
    @torch.no_grad() # we do not need to save gradients during validation
    def validate(self, loader, model=None, phase_name='val', **kwargs):
        """
        Validation process method, that estimates the performance of self.model on validation data in loader.
        
        Args:
            loader (DataLoader): validation set data loader to iterate over
            model (Model): torch nn.Module or nested class, that implements the model. Overwrites self.model.
            opt (Optimizer): torch optimizer to be used for loss minimization. Overwrites self.opt.
            **kwargs: additional parameters to pass to self.validate.
        """
        self._phase_name = phase_name
        self._reset_events(phase_name)
        self._run_epoch(loader, train_phase=False, output_log=True, **kwargs)
        return self.metrics




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

    test_batch_gen = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=BATCH_SIZE,
                                                num_workers=NUM_WORKERS)


    model = nn.Sequential()
    # reshape from "images" to flat vectors
    model.add_module('flatten', Flatten())
    # dense "head"
    model.add_module('dense1', nn.Linear(3 * SIZE_H * SIZE_W, 256))
    model.add_module('dense1_relu', nn.ReLU())
    model.add_module('dropout1', nn.Dropout(0.1))
    model.add_module('dense3', nn.Linear(256, EMBEDDING_SIZE))
    model.add_module('dense3_relu', nn.ReLU())
    model.add_module('dropout3', nn.Dropout(0.1))
    # logits for NUM_CLASSES=2: cats and dogs
    model.add_module('dense4_logits', nn.Linear(EMBEDDING_SIZE, NUM_CLASSES))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    ckpt_name = 'models/model_base.ckpt'
    model = model.to(device)

    runner = CNNRunner(model, opt, device, ckpt_name)
    
    runner.train(train_batch_gen, val_batch_gen, n_epochs=1, visualize=False)

