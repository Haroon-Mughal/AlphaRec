import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.abstract_model import AbstractModel
from .base.abstract_RS import AbstractRS
from .base.abstract_data import AbstractData, helper_load, helper_load_train
from tqdm import tqdm

from .base.evaluator import ProxyEvaluator
from .base.utils import *

from functools import partial

class AlphaRec_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)

    def train_one_epoch(self, epoch):
        running_loss, num_batches = 0, 0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for batch_i, batch in pbar:          
            
            batch = [x.to(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop  = batch[0], batch[1], batch[2], batch[3]

            if self.args.infonce == 0 or self.args.neg_sample != -1:
                neg_items = batch[4]
                neg_items_pop = batch[5]
            elif self.args.infonce == 1 and self.args.neg_sample == -1      # in-batch negatuve supcon case 
                neg_items = pos_items
            
            self.model.train()

            loss = self.model(users, pos_items, neg_items)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            num_batches += 1

        return [running_loss/num_batches]
    
class AlphaRec_Data(AbstractData):
    def __init__(self, args):
        super().__init__(args)
    
    def add_special_model_attr(self, args):
        self.lm_model = args.lm_model
        loading_path = args.data_path + args.dataset + '/item_info/'
        embedding_path_dict = {
            'bert': 'item_cf_embeds_bert_array.npy',
            'roberta': 'item_cf_embeds_roberta_array.npy',
            'v2': 'item_cf_embeds_array.npy',
            'v3': 'item_cf_embeds_large3_array.npy',
            'v3_shuffle': "item_cf_embeds_large3_array_shuffle.npy",
            'llama2_7b': 'item_cf_embeds_llama2_7b_array.npy',
            'llama3_7b': 'item_cf_embeds_llama3_7b_instruct_array.npy',
            'mistral_7b': 'item_cf_embeds_Norm_Mistral-7B-v0.1_array.npy',
            'SFR': 'item_cf_embeds_Norm_SFR-Embedding-Mistral_7b_array.npy',
            'GritLM_7b': 'item_cf_embeds_Norm_GritLM-7B_array.npy',
            'e5_7b': 'item_cf_embeds_Norm_e5-mistral-7b-instruct_array.npy',
            'echo_7b': 'item_cf_embeds_Norm_echo-mistral-7b-instruct-lasttoken_array.npy',
        }
        self.item_cf_embeds = np.load(loading_path + embedding_path_dict[self.lm_model])

        def group_agg(group_data, embedding_dict, key='item_id'):
            ids = group_data[key].values
            embeds = [embedding_dict[id] for id in ids]
            embeds = np.array(embeds)
            return embeds.mean(axis=0)

        # self.train_user_list
        pairs = []
        for u, v in self.train_user_list.items():
            for i in v:
                pairs.append((u, i))
        pairs = pd.DataFrame(pairs, columns=['user_id', 'item_id'])
        
        # User CF Embedding: the average of item embeddings
        groups = pairs.groupby('user_id')
        item_cf_embeds_dict = {i:self.item_cf_embeds[i] for i in range(len(self.item_cf_embeds))}
        user_cf_embeds = groups.apply(group_agg, embedding_dict=item_cf_embeds_dict, key='item_id')
        user_cf_embeds_dict = user_cf_embeds.to_dict()
        user_cf_embeds_dict = dict(sorted(user_cf_embeds_dict.items(), key=lambda item: item[0]))

        self.user_cf_embeds = np.array(list(user_cf_embeds_dict.values()))


import torch
import torch.nn.functional as F

def supcon_loss(user_emb, pos_item_embs, neg_item_embs, mask, tau, neg_sample):
    """
    Unified SupCon loss for both external and in-batch negative sampling modes.

    Args:
        user_emb:        [B, D]        - anchor user embeddings
        pos_item_embs:   [B, P, D]     - positive item embeddings (padded)
        neg_item_embs:   [B, N, D] or [B, P, D] - either sampled negatives or reused positives (for in-batch)
        mask:            [B, P]        - binary mask for valid positives
        tau:             float         - temperature
        neg_sample:      int           - if -1, use in-batch negatives; else use neg_item_embs

    Returns:
        Scalar SupCon loss
    """
    # Normalize all embeddings
    user_emb = F.normalize(user_emb, dim=-1)           # [B, D]
    pos_item_embs = F.normalize(pos_item_embs, dim=-1) # [B, P, D]
    neg_item_embs = F.normalize(neg_item_embs, dim=-1) # shape depends on mode

    B, P, D = pos_item_embs.shape
    user_exp = user_emb.unsqueeze(1).expand(-1, P, -1)  # [B, P, D]

    # Positive similarities: [B, P]
    pos_sim = torch.exp(torch.sum(user_exp * pos_item_embs, dim=-1) / tau)

    if neg_sample == -1:
        # ---------- IN-BATCH NEGATIVE SAMPLING ----------
        # Flatten all positive items across batch
        all_items_flat = neg_item_embs.view(B * P, D)             # [B*P, D]
        #all_items_flat = all_items_flat.detach()                  # optional: prevent gradients through negs

        # Similarities: [B, B*P]
        sim_matrix = torch.matmul(user_emb, all_items_flat.T) / tau
        sim_matrix = torch.exp(sim_matrix)

        # Create mask to exclude each user's own positives from denominator
        neg_mask = torch.ones((B, B * P), device=user_emb.device)
        for i in range(B):
            valid_p = int(mask[i].sum().item())
            neg_mask[i, i*P : i*P + valid_p] = 0  # zero out self-positives

        # Denominator: sum over other users' positives
        neg_denom = (sim_matrix * neg_mask).sum(dim=1, keepdim=True)  # [B, 1]
        denom = neg_denom + pos_sim  # [B, P] - broadcast adds back self-positives

    else:
        # ---------- EXTERNAL NEGATIVE SAMPLING ----------
        # Compute [B, N] similarities between users and their negatives
        neg_sim = torch.exp(
            torch.bmm(user_emb.unsqueeze(1), neg_item_embs.transpose(1, 2)).squeeze(1) / tau
        )  # [B, N]

        neg_sum = neg_sim.sum(dim=1, keepdim=True)    # [B, 1]
        denom = pos_sim + neg_sum.expand(-1, P)       # [B, P]

    # Compute log-probabilities for positives
    log_prob = torch.log(pos_sim / (denom + 1e-8))    # [B, P]

    # Apply mask to ignore padding
    masked_log_prob = log_prob * mask                # [B, P]
    user_loss = -masked_log_prob.sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [B]

    return user_loss.mean()



class AlphaRec(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.tau = args.tau
        self.embed_size = args.hidden_size
        self.lm_model = args.lm_model
        self.model_version = args.model_version
        self.neg_sample = args.neg

        self.init_user_cf_embeds = data.user_cf_embeds
        self.init_item_cf_embeds = data.item_cf_embeds

        self.init_user_cf_embeds = torch.tensor(self.init_user_cf_embeds, dtype=torch.float32).to(self.device)
        self.init_item_cf_embeds = torch.tensor(self.init_item_cf_embeds, dtype=torch.float32).to(self.device)

        self.init_embed_shape = self.init_user_cf_embeds.shape[1]
        
        # To keep the same parameter size
        multiplier_dict = {
            'bert': 8,
            'roberta': 8,
            'v2': 2,
            'v3': 1/2,
            'v3_shuffle': 1/2,
        } 
        if(self.lm_model in multiplier_dict):
            multiplier = multiplier_dict[self.lm_model]
        else:
            multiplier = 9/32 # for dimension = 4096

        if(self.model_version == 'homo'): # Linear mapping
            self.mlp = nn.Sequential(
            nn.Linear(self.init_embed_shape, self.embed_size, bias = False) # homo
            )

        else: # MLP
            self.mlp = nn.Sequential(
                nn.Linear(self.init_embed_shape, int(multiplier * self.init_embed_shape)),
                nn.LeakyReLU(),
                nn.Linear(int(multiplier * self.init_embed_shape), self.embed_size)
            )

    def init_embedding(self):
        pass


    def compute(self):
        users_cf_emb = self.mlp(self.init_user_cf_embeds)
        items_cf_emb = self.mlp(self.init_item_cf_embeds)

        users_emb = users_cf_emb
        items_emb = items_cf_emb

        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.data.n_users, self.data.n_items])
        
        return users, items

    def forward(self, users, pos_items, neg_items):

        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        if(self.train_norm):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)
        
        if(self.args.infonce == 1):
           pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
           neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), neg_emb.permute(0, 2, 1)).squeeze(dim=1)

           numerator = torch.exp(pos_ratings / self.tau)

           denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim = 1)
        
           ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        #  optonal SupCon Loss 
        supcon_loss_value = 0.0
        if self.args.use_supcon:
           # Build user history embeddings and mask
           pos_item_lists = [self.data.train_user_list[u.item()] for u in users]
           max_len = max(len(l) for l in pos_item_lists)

           padded = torch.zeros(len(pos_item_lists), max_len, dtype=torch.long, device=users.device)
           mask = torch.zeros_like(padded, dtype=torch.float)

           for i, item_ids in enumerate(pos_item_lists):
               padded[i, :len(item_ids)] = torch.tensor(item_ids, dtype=torch.long, device=users.device)
               mask[i, :len(item_ids)] = 1.0

           pos_item_embs = all_items[padded]  # [B, P, D]

           supcon_loss_value = supcon_loss(users_emb, pos_item_embs, neg_emb, mask, self.tau, self.neg_sample)     
            

        if self.args.combine_loss:
            return ssm_loss + self.args.supcon_weight * supcon_loss_value
        elif self.args.use_supcon:
            return supcon_loss_value
        else:
            return ssm_loss

    @torch.no_grad()
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.data.n_items))

        all_users, all_items = self.compute()
        
        users = all_users[torch.tensor(users).to(self.device)]
        items = all_items[torch.tensor(items).to(self.device)]
        
        if(self.pred_norm == True):
            users = F.normalize(users, dim = -1)
            items = F.normalize(items, dim = -1)
        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items) # user * item

        return rate_batch.cpu().detach().numpy()


