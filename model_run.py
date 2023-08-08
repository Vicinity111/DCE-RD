from sampler import Sampler
import torch
import torch.nn as nn
from torch.nn import LazyBatchNorm1d, ModuleList
from tqdm import tqdm
import torch.nn.functional as F
from utils import Criterion, EarlyStopping
import numpy as np
from sklearn.metrics import roc_auc_score
from torchmetrics import F1Score
from evaluate import evaluation_2class


class model_run(object):
  def __init__(self, model, extractor, optimizer, loaders, sampler_config, model_config, num_class, batch_size, device):
    super(model_run, self).__init__()
    
    self.device = device
    self.sampler_model=Sampler(sampler_config ,batch_size, device)
    self.attn_model = model
    self.extractor = extractor
    self.loaders=loaders
    self.batch_size=batch_size

    self.epochs=model_config["epochs"]
    self.Gnum_m=int(sampler_config["Gnum_m"])
    
    self.num_class=num_class
    self.multi_label=model_config['multi_label']
    self.learn_edge_att = model_config['learn_edge_att']
    self.criterion = Criterion(self.num_class,self.multi_label)

    self.lr_decay_factor = float(model_config.get('lr_decay_factor',0.5))
    self.lr_decay_step = int(model_config.get('lr_decay_step',30))
    self.split_way = sampler_config['edge_split']
    self.pred_coef = float(sampler_config["pred_coef"])
    self.sampler_coef = float(sampler_config["sampler_coef"])
    self.counter_coef = float(sampler_config["counter_coef"])
    self.optimizer_attn = optimizer
    

  def forward_pass(self, data, sub_X_features, sub_atts, att_scores):
    datalen = len(data[0])
    if self.num_class==2:
      clf_logits = torch.zeros(datalen, 1).unsqueeze(0).to(self.device)
      for index, subgraph in enumerate(data):
        att = sub_atts[index]
        edge_att = None
        if len(subgraph.edge_index)!=0:
          if self.learn_edge_att:
            edge_att = att
          else:
            edge_att = self.lift_node_att_to_edge_att(att, subgraph.edge_index)
        
        clf_logit = self.attn_model(sub_X_features[index], subgraph.edge_index, subgraph.batch, edge_attr=subgraph.edge_attr, edge_atten=edge_att)
        clf_logits = torch.cat((clf_logits, clf_logit.unsqueeze(0)),0)
      clf_logits_f = torch.zeros(datalen, 1).unsqueeze(0).to(self.device)
    
      for hj in range(self.Gnum_m):
        clf_logits_f = torch.cat((clf_logits_f,torch.mul(att_scores[hj].reshape(-1,1), clf_logits[hj+1].unsqueeze(0))),0)
    
    else:
      clf_logits = torch.zeros(datalen, self.num_class).unsqueeze(0).to(self.device)
      for index, subgraph in enumerate(data):
        att = sub_atts[index]
        edge_att = None
        if len(subgraph.edge_index)!=0:
          if self.learn_edge_att:
            edge_att = att
          else:
            edge_att = self.lift_node_att_to_edge_att(att, subgraph.edge_index)
        

        clf_logit = self.attn_model(subgraph.x.float(), subgraph.edge_index, subgraph.batch, edge_attr=subgraph.edge_attr, edge_atten=edge_att)
        clf_logits = torch.cat((clf_logits, clf_logit.unsqueeze(0)),0)
      clf_logits_f = torch.zeros(datalen, self.num_class).unsqueeze(0).to(self.device)
    
      for hj in range(self.Gnum_m):
        clf_logits_f = torch.cat((clf_logits_f,torch.mul(att_scores[hj].reshape(-1,1), clf_logits[hj+1].unsqueeze(0))),0)
    clf_logits_f = clf_logits_f.sum(0)
    pred_loss = self.criterion(clf_logits_f, data[0].y)
    return pred_loss, clf_logits_f
     

  def forward(self, data, sub_X_features, sub_atts, epoch, training):

    att_scores = torch.tensor([]).to(self.device)
    sloss_total = []
    for i in range(len(data[0])):
      att_score, sloss = self.sampler_model.dpp_computeLoss(i)
      att_scores = torch.cat((att_scores, att_score.T),1)
      sloss_total.append(sloss)

    pred_loss, clf_logits_f = self.forward_pass(data, sub_X_features, sub_atts,  att_scores)

    sampler_loss = torch.mean(torch.tensor(sloss_total).to(self.device))
    return pred_loss, sampler_loss, clf_logits_f
  
  def batch_split(self, embeddings, i):
    subGraph_embs = []
  
    for sub in embeddings:
      subGraph_embs.append(sub.index_select(0, torch.tensor(np.arange(self.sub_split[i], self.sub_split[i+1]))))
    return subGraph_embs


  @staticmethod
  def lift_node_att_to_edge_att(node_att, edge_index):
      src_lifted_att = node_att[edge_index[0]]
      dst_lifted_att = node_att[edge_index[1]]
      edge_att = src_lifted_att * dst_lifted_att
      return edge_att.sum(1, keepdim = True)
  
  @staticmethod
  def get_preds(logits, multi_label):
    if multi_label:
        preds = (logits.sigmoid() > 0.5).float()
    elif logits.shape[1] > 1:  # multi-class
        preds = logits.argmax(dim=1).float().unsqueeze(dim=1)
    else:  # binary
        preds = (logits.sigmoid() > 0.5).float()
    return preds

  @torch.no_grad()
  def eval_one_batch(self, data, sub_X_features, counter_loss, sub_atts, pred_loss2, epoch):
      self.extractor.eval()
      self.attn_model.eval()

      pred_loss1, sampler_loss, clf_logits_f = self.forward(data, sub_X_features, sub_atts, epoch, training=False)
      pred_loss = pred_loss1 + pred_loss2
      loss =  self.pred_coef * pred_loss + self.sampler_coef * sampler_loss + counter_loss * self.counter_coef

      return loss,  clf_logits_f

  def train_one_batch(self, data, sub_X_features, counter_loss, sub_atts, pred_loss2, epoch):
      self.extractor.train()
      self.attn_model.train()

      pred_loss1, sampler_loss, clf_logits_f = self.forward(data, sub_X_features, sub_atts, epoch, training=True)
      pred_loss = pred_loss1 + pred_loss2
      loss =  self.pred_coef * pred_loss + self.sampler_coef * sampler_loss + counter_loss * self.counter_coef

      self.attn_model.train()

      self.optimizer_attn.zero_grad()
      loss.backward()
      self.optimizer_attn.step()
      return loss, clf_logits_f

  def get_ori(self, phase, data):
    if phase=="train":
      self.extractor.train()
      self.attn_model.train()

      ori_embs, X_feature = self.attn_model.get_emb(data.x.float(), data.edge_index, batch=data.batch, edge_attr=data.edge_attr.float(), sentence_tokens=data.sentence_tokens)
      att_log_logit = self.extractor(ori_embs, data.edge_index, data.batch)
      att = (att_log_logit).sigmoid()
      if self.learn_edge_att:
        edge_att = att
      else:
        edge_att = self.lift_node_att_to_edge_att(att, data.edge_index).to(self.device)
     
      clf_logit = self.attn_model(X_feature, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att, sentence_tokens=data.sentence_tokens)
      pred_loss2 = self.criterion(clf_logit, data.y)
      pred_loss_ori = pred_loss2 
    else:
      self.extractor.eval()
      self.attn_model.eval()

      ori_embs, X_feature = self.attn_model.get_emb(data.x.float(), data.edge_index, batch=data.batch, edge_attr=data.edge_attr.float(), sentence_tokens=data.sentence_tokens)
      att_log_logit = self.extractor(ori_embs, data.edge_index, data.batch)
      att = (att_log_logit).sigmoid()
      if self.learn_edge_att:
        edge_att = att
      else:
        edge_att = self.lift_node_att_to_edge_att(att, data.edge_index).to(self.device)
     
      clf_logit = self.attn_model(X_feature, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att, sentence_tokens=data.sentence_tokens)
      pred_loss2 = self.criterion(clf_logit, data.y) 
      pred_loss_ori = pred_loss2 
    return ori_embs, att, pred_loss_ori, X_feature

  def model_run_one_epoch(self, data_loader, epoch, phase, dataname):

    datalen=len(data_loader.dataset)

    run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
    phase = 'test ' if phase == 'test' else phase
    
    pbar = tqdm(data_loader)

    total_acc = 0
    f1_score = []
    total_auc = 0
    probs_all = []
    label_all = []
    tloss = 0
    idx = 0

    Prec1_all, Recll1_all, F1_all,Prec2_all, Recll2_all, F2_all= [],[],[],[],[],[]
    for idx, data in enumerate(pbar):
      
      data = data.to(self.device)
      ori_embs, att, pred_loss_ori, X_feature = self.get_ori(phase, data)
      sub_graphs, sub_atts, counter_sub_graphs_idxs, sub_X_features= self.sampler_model(data, ori_embs, att, X_feature)

      counter_loss = 0
      for i in range(self.Gnum_m):
        masked_emb = torch.zeros_like(ori_embs)
        masked_emb[counter_sub_graphs_idxs[i]] = ori_embs[counter_sub_graphs_idxs[i]]
        counter_clf_logit = self.attn_model.get_pred_from_emb(masked_emb, data.batch)
        #closs_ii = self.criterion(counter_clf_logit, data.y)
        closs_ij = self.criterion(counter_clf_logit, torch.logical_xor(torch.ones_like(data.y), data.y))
        counter_loss += closs_ij

      counter_loss = counter_loss / self.Gnum_m
      loss, clf_logits_f = run_one_batch(sub_graphs, sub_X_features, counter_loss, sub_atts, pred_loss_ori, epoch)
      clf_preds = self.get_preds(clf_logits_f, self.multi_label)

      _, Prec1, Recll1, F1, Prec2, Recll2, F2 = evaluation_2class(clf_preds, data.y)
    
      Prec1_all.append(Prec1)
      Recll1_all.append(Recll1)
      F1_all.append(F1)
      Prec2_all.append(Prec2)
      Recll2_all.append(Recll2)
      F2_all.append(F2)
      
      total_acc += (clf_preds == data.y).sum().item()

      f1 = F1Score(num_classes=self.num_class, average="weighted").to(self.device)
      f1_score.append(f1(clf_preds.int(), data.y.int()).item())   

      if self.num_class==2:
        probs = F.sigmoid(clf_logits_f)
      else:
        probs = F.softmax(clf_logits_f)
      probs_all.extend(probs.cpu().detach().tolist())
      label_all.extend(data.y.squeeze(1).cpu().detach().numpy())

      tloss += loss.item()

    if "Twitter" in dataname:
      total_auc = roc_auc_score(label_all, np.array(probs_all),average="weighted", multi_class='ovr')
    else:
      total_auc = roc_auc_score(label_all, np.array(probs_all))

    return total_acc / datalen, np.mean(f1_score), np.mean(Prec1_all), np.mean(Recll1_all), np.mean(F1_all), np.mean(Prec2_all), np.mean(Recll2_all), np.mean(F2_all), tloss/(1+idx), total_auc

      
  def model_train(self, dataname):
      
      best_test_acc, best_test_acc_epoch, best_test_f1, best_test_f1_epoch, best_test_auc , best_test_auc_epoch =0,0,0,0,0,0
      early_stopping = EarlyStopping(patience=10, verbose=True)

      for epoch in range(self.epochs):
        acc_train, F1_train, prec1_train, recal1_train, f1_train, prec2_train, recal2_train, f2_train, loss_train, train_auc \
          = self.model_run_one_epoch(self.loaders['train'], epoch, "train", dataname)
        acc_test, F1_test, prec1_test, recal1_test, f1_test, prec2_test, recal2_test, f2_test, loss_test, test_auc \
          = self.model_run_one_epoch(self.loaders['test'], epoch, "test", dataname)  
        if best_test_auc<test_auc:
          best_test_auc , best_test_auc_epoch = test_auc, epoch
        if best_test_acc<acc_test:
          best_test_acc , best_test_acc_epoch = acc_test, epoch
        if best_test_f1<F1_test:
          best_test_f1 , best_test_f1_epoch = F1_test, epoch
        early_stopping(loss_test, acc_test, F1_test, test_auc, 'DCE-RD', dataname)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        print(f'[Epoch: {epoch}], Train_LOSS: {loss_train: .4f}, Train Pred F1: {F1_train: .4f}, Train Pred ACC: {acc_train: .4f}, Train AUC: {train_auc: .4f}',"\n",
        f'Train Prec1: {prec1_train: .4f}, Train Pred2: {prec2_train: .4f}, Train Recall1: {recal1_train: .4f}, Train Recall2: {recal2_train: .4f}',"\n",
            f'Train F1_0: {f1_train: .4f}, Train F1_1: {f2_train: .4f}')
        print(f'[Epoch: {epoch}], Test_LOSS: {loss_test: .4f}, Test Pred F1: {F1_test: .4f}, Test Pred ACC: {acc_test: .4f}, Test AUC: {test_auc: .4f}',"\n",
         f'Test Prec1: {prec1_test: .4f}, Test Pred2: {prec2_test: .4f}, Test Recall1: {recal1_test: .4f}, Test Recall2: {recal2_test: .4f}',"\n",
            f'Test F1_0: {f1_test: .4f}, Test F1_1: {f2_test: .4f}')
        print('====================================')
        print('====================================')  
