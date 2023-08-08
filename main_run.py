from utils import get_data_loaders, get_model, MLP
import yaml
from pathlib import Path
import torch
import torch.nn as nn

from model_run import model_run

import warnings
warnings.simplefilter("ignore", UserWarning)

class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, model_config, Gnum_m):
        super().__init__()
        self.learn_edge_att = model_config['learn_edge_att']
        dropout_p = model_config['extractor_dropout_p']

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, Gnum_m], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits

if __name__=="__main__":
    local_config = yaml.safe_load((Path("./configs/DCE-RD_Graph_MCFake.yml")).open('r'))

    model_config = local_config['model_config']
    sampler_config = local_config["sampler_config"]
    data_config= local_config["data_config"]
    data_nam = "Graph_MCFake"
    
    
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(
        data_dir="./data/", dataset_name=data_nam,batch_size=data_config['batch_size'], splits = None, random_state=0)
    
    model_config['deg'] = aux_info['deg']
    
    cuda_id = 5 #or -1 if cpu

    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')
    
    print("[INFO] Dataset:",  data_nam)
    print("[INFO] Model:",  model_config["model_name"])
    print("[INFO] Using device:",  device)
    print("[INFO] lr:",  model_config["lr"])
    print("[INFO] batch-size:",  data_config["batch_size"])
    print("[INFO] Gnum:" , sampler_config["Gnum_m"])
    print("[INFO] Knum: ", sampler_config["Nnum_k"])
    print("[INFO] hidden-size:  ", model_config["hidden_size"])
    print("[INFO] pred-coef:  ",sampler_config["pred_coef"])
    print("[INFO] sampler-coef:" ,sampler_config["sampler_coef"])
    print("[INFO] counter-coef:",sampler_config["counter_coef"])

    for i in loaders.keys():
        print("--------5dold: ", i, " ---------------")
        model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)
        extractor=ExtractorMLP(model_config['hidden_size'], model_config, sampler_config["Gnum_m"]).to(device)
        optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=float(model_config["lr"]), weight_decay=float(model_config["weight_decay"]))
        
        trainer=model_run(model, extractor, optimizer, loaders[i], sampler_config, model_config, num_class, data_config["batch_size"], device)
        trainer.model_train(data_nam)

    
      

      
