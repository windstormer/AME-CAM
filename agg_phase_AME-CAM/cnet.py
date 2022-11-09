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

class CNet(object):
    def __init__(self, args, train_loader, val_loader, log_path, record_path, model_name, gpuid):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_path = log_path
        
        model = Res18_Classifier()

        score_model = Res18_Scoring()

        if args.encoder_pretrained_path != None:
            encoder_pretrained_path = os.path.join(args.project_path, "record/CNet", args.encoder_pretrained_path, "model", "model.pth")
            model.load_pretrain_weight(encoder_pretrained_path)

        if args.pretrained_path != None:
            pretrained_path = os.path.join(args.project_path, "record/CNet", args.pretrained_path, "model", "model.pth")
            model.load_pretrain_weight(pretrained_path)
            pretrained_path = os.path.join(args.project_path, "record/CNet", args.pretrained_path, "model", "score_model.pth")
            score_model.load_pretrain_weight(pretrained_path)



        self.lr = args.learning_rate
        # self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.score_optimizer = optim.Adam(score_model.parameters(), lr=self.lr, weight_decay=1e-5)
        # self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.score_optimizer, T_max=args.epochs, eta_min=0.000005)
        
        for param in model.parameters():
            param.requires_grad = False
        if len(gpuid) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpuid)
            score_model = torch.nn.DataParallel(score_model, device_ids=gpuid)
        # pretrain_model_path = os.path.join(project_path, record_path, "weights", "embedder.pth")
        self.model = model.to('cuda')
        self.score_model = score_model.to('cuda')

        self.loss = nn.BCEWithLogitsLoss()
        self.sminloss = SimMinLoss()
        self.smaxloss = SimMaxLoss()
        self.model_name = model_name
        self.project_path = args.project_path
        self.record_path = record_path

    def run(self):
        log_file = open(self.log_path, "w+")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        train_record = {'auc':[], 'loss':[]}
        val_record = {'auc':[], 'loss':[]}
        best_score = 0.0
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc, sensitivity, specificity, train_auc = self.train(epoch)
            train_record['loss'].append(train_loss)
            train_record['auc'].append(train_auc)

            self.scheduler.step()
            
            log_file = open(self.log_path, "a")
            log_file.writelines(
            f'Epoch {epoch:4d}/{self.epochs:4d} | Cur lr: {self.scheduler.get_last_lr()[0]} | Train Loss: {train_loss}, Train Acc: {train_acc}, AUC: {train_auc}\n')

            val_loss, val_acc, sensitivity, specificity, val_auc = self.val(epoch)
            val_record['loss'].append(val_loss)
            val_record['auc'].append(val_auc)
            log_file = open(self.log_path, "a")
            log_file.writelines(
            f"Epoch {epoch:4d}/{self.epochs:4d} | Val Loss: {val_loss}, Val Acc: {val_acc}, AUC: {val_auc}, Sensitivity: {sensitivity}, Specificity: {specificity}\n")

            # cur_score = val_auc
            # if cur_score > best_score:
            #     best_score = cur_score
            #     log_file.writelines(f"Save model at Epoch {epoch:4d}/{self.epochs:4d} | Val Loss: {val_loss}, Val Acc: {val_acc}, AUC: {val_auc}\n")
            model_path = os.path.join(self.project_path, self.record_path, self.model_name, "model", "model.pth")
            torch.save(self.model.state_dict(), model_path)
            model_path = os.path.join(self.project_path, self.record_path, self.model_name, "model", "score_model.pth")
            torch.save(self.score_model.state_dict(), model_path)
            log_file.close()

            # parameter_path = os.path.join(self.project_path, self.record_path, self.model_name, "model", "model.pth")
            # torch.save(self.model.state_dict(), parameter_path)
            log_file = open(self.log_path, "a")
            log_file.writelines(str(datetime.now())+"\n")
            log_file.close()
        save_chart(self.epochs, train_record['auc'], val_record['auc'], os.path.join(self.project_path, self.record_path, self.model_name, "auc.png"), name='auc')
        save_chart(self.epochs, train_record['loss'], val_record['loss'], os.path.join(self.project_path, self.record_path, self.model_name, "loss.png"), name='loss')

    def train(self, epoch):
        self.model.eval()
        self.score_model.train()
        train_bar = tqdm(self.train_loader)
        total_loss, total_num = 0.0, 0
        train_labels = []
        pred_results = []
        log_file = open(self.log_path, "a")
        

        for case_batch, label_batch in train_bar:
            
            # self.model.train()
            # self.optimizer.zero_grad()
            celoss, pred_batch, celoss_collect, map_collect = self.step(case_batch, label_batch)
            # celoss.backward()
            # self.optimizer.step()
            # self.model.eval()
            self.score_optimizer.zero_grad()
            clloss, clloss_collect = self.score_step(case_batch, map_collect)
            clloss.backward()
            self.score_optimizer.step()


            total_num += self.batch_size
            total_loss += (clloss.item()) * self.batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.epochs, total_loss / total_num))
            
            pred_results.append(pred_batch)
            train_labels.append(label_batch)
            print("Train Loss:")
            # log_file.writelines('Train Loss:\n')
            for k, v in {**celoss_collect, **clloss_collect}.items():
                print(f'\t{k}: {v.item()}')
                # log_file.writelines(f'\t{k}: {v.item()}\n')
            
        log_file.close()

        pred_results = torch.cat(pred_results, dim=0).numpy()
        train_labels = torch.cat(train_labels, dim=0).numpy()
        # print(pred_results.shape)
        # print(train_labels.shape)
        acc, sensitivity, specificity, auc_score = self.evaluate(train_labels, pred_results)
        return total_loss / total_num, acc, sensitivity, specificity, auc_score

    def step(self, data_batch, label_batch):
        logits_collect, map_collect = self.model(data_batch.cuda())
        loss = 0
        # ic_weight = [0.25, 0.5, 0.75, 1.0]
        loss_collect = {}
        for idx, logits in enumerate(logits_collect):
            loss_collect[f"ic_{idx}"] = (self.loss(logits, label_batch.cuda()))

        for k, l in loss_collect.items():
            loss += l
        pred = torch.sigmoid(logits_collect[-1])
        return loss, pred.detach().cpu(), loss_collect, map_collect
    
    def score_step(self, data_batch, map_collect):
        final_map, foreground, background = self.score_model(data_batch.cuda(), map_collect)
        loss = 0
        # ic_weight = [0.25, 0.5, 0.75, 1.0]
        loss_collect = {}

        loss_collect["SimMax_Foreground"] = (self.smaxloss(foreground))
        loss_collect["SimMin"] = (self.sminloss(foreground, background))
        loss_collect["SimMax_Background"] = (self.smaxloss(background))
        for k, l in loss_collect.items():
            loss += l

        return loss, loss_collect

            
    def val(self, epoch):
        self.model.eval()
        self.score_model.eval()
        val_bar = tqdm(self.val_loader)
        total_loss, total_num = 0.0, 0
        val_labels = []
        pred_results = []
        out_results = []
        with torch.no_grad():
            for case_batch, label_batch in val_bar:
                celoss, pred_batch, loss_collect, map_collect = self.step(case_batch, label_batch)
                clloss, loss_collect = self.score_step(case_batch, map_collect)

                total_num += self.batch_size
                total_loss += (celoss.item()+clloss.item()) * self.batch_size
                val_bar.set_description('Val Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.epochs, total_loss / total_num))
                pred_results.append(pred_batch)
                val_labels.append(label_batch)
        pred_results = torch.cat(pred_results, dim=0).numpy()
        val_labels = torch.cat(val_labels, dim=0).numpy()
        # print(pred_results.shape)
        # print(val_labels.shape)
        acc, sensitivity, specificity, auc_score = self.evaluate(val_labels, pred_results)
        return total_loss / total_num, acc, sensitivity, specificity, auc_score

    def test(self, loader, load_model=None):
        self.model.eval()
        self.score_model.eval()
        test_bar = tqdm(loader)
        total_loss, total_num = 0.0, 0
        test_labels = []
        pred_results = []
        with torch.no_grad():
            for case_batch, label_batch in test_bar:
                celoss, pred_batch, loss_collect, map_collect = self.step(case_batch, label_batch)
                clloss, loss_collect = self.score_step(case_batch, map_collect)

                total_num += self.batch_size
                total_loss += (celoss.item()+clloss.item()) * self.batch_size
                pred_results.append(pred_batch)
                test_labels.append(label_batch)
        pred_results = torch.cat(pred_results, dim=0).numpy()
        test_labels = torch.cat(test_labels, dim=0).numpy()
        # print(pred_results.shape)
        # print(test_labels.shape)
        acc, sensitivity, specificity, auc_score = self.evaluate(test_labels, pred_results)
        return total_loss / total_num, acc, sensitivity, specificity, auc_score

    def tensor_random_sample_idx(self, idx, p=0.05):
        idx_len = len(idx)
        rand_idx = torch.randperm(idx_len)
        new_idx = idx[rand_idx]
        sample = int(idx_len*p)
        return new_idx[:sample]


    def evaluate(self, labels, pred):
        # fpr, tpr, threshold = roc_curve(labels, pred, pos_label=1)
        # fpr_optimal, tpr_optimal, threshold_optimal = self.optimal_thresh(fpr, tpr, threshold)
        out_results = [pred > 0.5 for pred in pred]
        auc_score = roc_auc_score(labels, pred)

        tn, fp, fn, tp = confusion_matrix(labels, out_results, labels=[0,1]).ravel()
        acc = (tp+tn) / (tn+fp+fn+tp)
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp+fn)
        return acc, sensitivity, specificity, auc_score

    # def optimal_thresh(self, fpr, tpr, thresholds, p=0):
    #     loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    #     idx = np.argmin(loss, axis=0)
    #     return fpr[idx], tpr[idx], thresholds[idx]