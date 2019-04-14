import numpy as np
import torch
from torchvision.utils import make_grid
import random
from base import BaseTrainer
import seaborn as sns
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


from collections import deque


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.L = config['trainer']['L']
        self.max_pen_move = config['trainer']['max_pen_move']

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def getDistanceMap(self, x, y, L):
        distanceMap = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                distanceMap[i, j] = np.sqrt((i - x) ** 2 + (j - y) ** 2) / L
        return distanceMap

    def getSubCanvas(self, x, y, canvas):
        return canvas[x - self.max_pen_move:x + self.max_pen_move + 1, y - self.max_pen_move:y + self.max_pen_move + 1]

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
    
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        pen_position = (15,15)
        pen_state = 1
        canvas = np.zeros((self.L,self.L))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            trsfm = transforms.Compose([
                transforms.ToTensor(),
            ])
            im = data[0,0,:,:].numpy()
            distanceMap = self.getDistanceMap(pen_position[0], pen_position[1], self.L)
            colormap = np.ones((self.L,self.L)) if pen_state == 1 else np.zeros((self.L,self.L))
            localCanvas = self.getSubCanvas(pen_position[0], pen_position[1], im)
            self.optimizer.zero_grad()
            global_data = np.stack([canvas,im,distanceMap,colormap])
            global_data = np.swapaxes(global_data,0,1)
            global_data = np.swapaxes(global_data, 1, 2)
            if trsfm:
                global_data = trsfm(global_data)
            global_data =  global_data.unsqueeze(0)
            local_data = localCanvas
            local_data = np.expand_dims(local_data, axis=0) #Need to add extra dimension to specify 1 channel
            local_data = np.swapaxes(local_data,0,1)
            local_data = np.swapaxes(local_data, 1, 2)
            if trsfm:
                local_data = trsfm(local_data)
            local_data = local_data.unsqueeze(0)
            #send data to shadow realm (GPU)
            global_data = global_data.to(self.device)
            local_data = local_data.to(self.device)
            output = self.model(global_data, local_data)
            #output = torch.unsqueeze(output,0)
            output = output.to(self.device)
            target = target.to(self.device)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
