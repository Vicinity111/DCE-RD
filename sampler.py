import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_scatter
from tqdm import tqdm
import copy

EPSILON = torch.tensor(torch.finfo(torch.float32).tiny)

def compute_diversity_loss(phi_sub, phi_input):
    def compute_diversity(phi):
        phi = f.normalize(phi, p=2, dim=1)
        S_B = torch.mm(phi, phi.t())
        eig_vals, eig_vecs = torch.linalg.eig(S_B)
        return eig_vals.float(), eig_vecs.float()

    def normalize_min_max(eig_vals):
        min_v, max_v = torch.min(eig_vals), torch.max(eig_vals)
        return (eig_vals - min_v + 1e-9) / (max_v - min_v + 1e-9)

    sub_eig_vals, sub_eig_vecs = compute_diversity(phi_sub)
    input_eig_vals, input_eig_vecs = compute_diversity(phi_input)
    magnitude_loss = torch.exp(-torch.norm(input_eig_vals-sub_eig_vals, p=2, dim=0, keepdim=True))
    weighted_structure_loss = torch.sum(torch.norm(torch.mul(sub_eig_vecs, input_eig_vecs), p=2, dim=1, keepdim=True))

    return magnitude_loss + weighted_structure_loss

class Sampler(nn.Module):

  def __init__(self, sampler_config, batch_size, device):
    super(Sampler,self).__init__()

    self.device =device
    self.Nnum_k=float(sampler_config["Nnum_k"])
    self.Gnum_m=int(sampler_config["Gnum_m"])
    self.temperature=float(sampler_config["temperature"])
    self.separate=sampler_config["separate"]
    self.batch_size = batch_size

  
  def forward(self, data, ori_embs, att, X_feature):
    self.data=data
    self.X_feature = X_feature
    
    nodes_num = torch.bincount(self.data.batch)
    self.nodes_num = torch.cat([torch.tensor([0]).to(self.device),torch.cumsum(nodes_num, 0)])
    self.subNnum_k = [int(np.ceil(d*self.Nnum_k)) for d in nodes_num.cpu().numpy()]
    self.ori_embs = ori_embs
    self.att = att
    self.subNodes = torch.cat([torch.tensor([0]).to(self.device),torch.cumsum(torch.tensor(self.subNnum_k).to(self.device), 0)])
    sub_graphs, counter_sub_graphs_idxs, sub_X_feature = self.graph_sampling(self.att)

    sub_atts = []
    for idxs in self.all_top_ks:
      sub_atts.append(self.att.index_select(0, torch.tensor(idxs).to(self.device)))

    return sub_graphs, sub_atts, counter_sub_graphs_idxs, sub_X_feature
  
  def dpp_computeLoss(self, embeddsub_idx):
      
      sub_dpps = []
      det_all = []
      info_loss = 0

      for m in range(self.Gnum_m):
        
        idxs = np.array(self.all_top_ks[m][self.subNodes[embeddsub_idx]:self.subNodes[embeddsub_idx+1]]) 
        L_sub = self.ori_embs.index_select(0, torch.tensor(idxs).to(self.device))
        sub_dpps.append(L_sub)
        det_all.append(torch.mean(L_sub, 0).cpu().detach().numpy())
      
      diversity = []

      for i in range(len(sub_dpps)):
        for j in range(len(sub_dpps)):
            if i!=j:
              diversity.append(compute_diversity_loss(sub_dpps[i], sub_dpps[j]) )

      loss =  torch.max(torch.tensor(diversity))  
      attn_score = self.subatt_score(sub_dpps) 
      return attn_score, loss.to(self.device)
  
  def subatt_score(self, embeddings):
     
      L = torch.tensor([]).to(self.device)
      for i in range(self.Gnum_m):     
          Ls = 0
          for j in range(self.Gnum_m):
            if i!=j:
              distance = torch.norm(embeddings[i]- embeddings[j], p=2, keepdim=True) ** 2 
              Ls += torch.log((distance + 1) / (distance + 1e-4))
          L = torch.cat((L, torch.sum(Ls).unsqueeze(0)), 0)  
      att_score = L / L.sum()

      return att_score.reshape(1,-1)

 
  def gumbel_keys(self, att):
    # sample some gumbels
    uniform = torch.rand_like(att)
    z = -torch.log(-torch.log(uniform + EPSILON))
    att_g = att + z
    return att_g


  def batch_softmax(self, att_g):
      exp_logits = torch.exp(torch.tensor(att_g) / self.temperature)
      partition = torch_scatter.scatter_sum(exp_logits, self.edges_batch, 0)
      partition = partition.index_select(0, self.edges_batch.T.squeeze())
      softmax_logits = exp_logits / (partition + EPSILON)
      return softmax_logits

  def continuous_topk(self, att_g, i):

    khot_list = torch.tensor([]).to(self.device)
    onehot_approx = torch.zeros_like(att_g)
    
    for _ in range(self.subNnum_k[i]):
        khot_mask = torch.maximum(1.0 - onehot_approx, EPSILON)
        att_g = att_g + torch.log(khot_mask)
        onehot_approx = nn.functional.softmax(att_g / self.temperature, 0)
        khot_list = torch.cat((khot_list,onehot_approx.T),dim=0)
    
    if self.separate:
        return khot_list
    else:
        return torch.sum(khot_list, 0)
  
  def reconstruct(self, top_k_idxs):
    # print(data.x.shape)
    data_copy=copy.deepcopy(self.data)
    data_copy.x=data_copy.x[top_k_idxs,:]

    edge_sel = []
    for idx,edge in enumerate(data_copy.edge_index.T):
      src = edge[0].item()
      dst = edge[1].item()
      if (src in top_k_idxs) and (dst in top_k_idxs):
        edge_sel.append(idx)
    data_copy.edge_attr=data_copy.edge_attr[edge_sel]
    data_copy.edge_index=data_copy.edge_index[:,edge_sel]
    data_copy.batch=data_copy.batch[top_k_idxs]
   
    top_k_idxs.sort()
    nodes_redict = {id:i for i, id in enumerate(top_k_idxs)}

    edges_list = [[nodes_redict[tup[0].item()], nodes_redict[tup[1].item()]] for tup in data_copy.edge_index.T]
    data_copy.edge_index=torch.from_numpy(np.array(edges_list).T).to(self.device)
    return data_copy

  def sample_subset(self, att, i):
    '''
    Args:
        att (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        i (int): index of batch
    '''
    att_g = self.gumbel_keys(att)
    return self.continuous_topk(att_g, i)

  def graph_sampling(self, att):

    sub_graphs=[0]*self.Gnum_m
    counter_sub_graphs_idxs=[]
    sub_X_features = []
    self.all_top_ks = []

    for j in range(self.Gnum_m):
      top_k_idxs = []
      counter_top_k_idxs = []
     
      for i in range(len(self.data)):
        selected_edges=self.sample_subset(att.index_select(1, torch.tensor(j).to(self.device)).index_select(0, torch.tensor(np.arange(self.nodes_num[i].cpu(), self.nodes_num[i+1].cpu())).to(self.device)), i)
        top_k_idx = np.sort(np.array(torch.argsort(selected_edges).cpu())[-self.subNnum_k[i]:])+self.nodes_num[i].item()
        counter_top_k_idx = np.sort(np.array(torch.argsort(selected_edges).cpu())[: -self.subNnum_k[i]])+self.nodes_num[i].item()
        top_k_idxs.extend(top_k_idx)
        counter_top_k_idxs.extend(counter_top_k_idx)
      
      sub_X_features.append(self.X_feature[top_k_idxs,:])
      self.all_top_ks.append(top_k_idxs)
      sub_graph = self.reconstruct(top_k_idxs)
      sub_graphs[j] = sub_graph
      counter_sub_graphs_idxs.append(counter_top_k_idxs)

    return sub_graphs, counter_sub_graphs_idxs, sub_X_features
