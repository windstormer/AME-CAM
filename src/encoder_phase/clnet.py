import torch
import torch.optim as optim
import os
from datetime import datetime
from collections import OrderedDict

from models import *
from loss import *
from tqdm import tqdm
from utils import *
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

class CLNet(object):
    def __init__(self, args, train_loader, val_loader, log_path, record_path, model_name, gpuid):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_path = log_path
        
        model = Res18()

        self.lr = args.learning_rate
        self.optimizer = optim.SGD(model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=0, last_epoch=-1)
        
        if len(gpuid) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpuid)
        # pretrain_model_path = os.path.join(project_path, record_path, "weights", "embedder.pth")
        self.model = model.to('cuda')

        self.loss = SupConLoss()
        self.model_name = model_name
        self.project_path = args.project_path
        self.record_path = record_path

    def run(self):
        log_file = open(self.log_path, "w+")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        record = {'train_loss':[], 'val_loss':[]}
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train(epoch)
            record['train_loss'].append(train_loss)

            if epoch >= 10:
                self.scheduler.step()
            
            log_file = open(self.log_path, "a")
            log_file.writelines(f'Epoch {epoch:4d}/{self.epochs:4d} | Cur lr: {self.scheduler.get_last_lr()[0]} | Train Loss: {train_loss}\n')

            val_loss = self.val(epoch)
            record['val_loss'].append(val_loss)
            log_file = open(self.log_path, "a")
            log_file.writelines(f"Epoch {epoch:4d}/{self.epochs:4d} | Val Loss: {val_loss}\n")
            log_file.close()

            parameter_path = os.path.join(self.project_path, self.record_path, self.model_name, "model", "encoder.pth")
            torch.save(self.model.state_dict(), parameter_path)
            log_file = open(self.log_path, "a")
            log_file.writelines(str(datetime.now())+"\n")
            log_file.close()
        save_chart(self.epochs, record['train_loss'], record['val_loss'], os.path.join(self.project_path, self.record_path, self.model_name, "loss.png"), name='loss')

    def train(self, epoch):
        self.model.train()
        train_bar = tqdm(self.train_loader)
        total_loss, total_num = 0.0, 0
        for aug1, aug2, label in train_bar:
            self.optimizer.zero_grad()
            loss = self.step(aug1, aug2, label)
            loss.backward()
            self.optimizer.step()

            total_num += self.batch_size
            total_loss += loss.item() * self.batch_size
            average_loss = (total_loss / total_num)
            train_bar.set_description(f'Train Epoch: [{epoch}/{self.epochs}] Loss: {average_loss:.4f}')

        return total_loss / total_num

    def step(self, aug1, aug2, label):
        aug1 = aug1.cuda()
        aug2 = aug2.cuda()
        feature_1, out_1 = self.model(aug1)
        feature_2, out_2 = self.model(aug2)

        # out_1 = F.normalize(out_1, dim=1)
        # out_2 = F.normalize(out_2, dim=1)
        clloss = self.loss(out_1, out_2, None)
        return clloss

            
    def val(self, epoch):
        self.model.eval()
        val_bar = tqdm(self.val_loader)
        total_loss, total_num = 0.0, 0
        with torch.no_grad():
            for aug1, aug2, label in val_bar:
                loss = self.step(aug1, aug2, label)

                total_num += self.batch_size
                total_loss += loss.item() * self.batch_size
                average_loss = (total_loss / total_num)
                val_bar.set_description(f'Val Epoch: [{epoch}/{self.epochs}] Loss: {average_loss:.4f}')
        return total_loss / total_num