import torch
from torch import nn
from torch.utils import data
import numpy as np
import tqdm

class TripleDataset(data.Dataset):

    def __init__(self, ent2id, rel2id, triple_data_list):
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.data = triple_data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        head, relation, tail = self.data[index]
        head_id = self.ent2id[head]
        relation_id = self.rel2id[relation]
        tail_id = self.ent2id[tail]
        return head_id, relation_id, tail_id
    
class GNN(nn.Module):

    def __init__(self, entity_num, relation_num, norm=1, dim=100):
        super(GNN, self).__init__()
        self.norm = norm
        self.dim = dim
        self.entity_num = entity_num
        self.entities_emb = self._init_emb(entity_num)
        self.relations_emb = self._init_emb(relation_num)

    def _init_emb(self, num_embeddings):
        embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.dim)
        uniform_range = 6 / np.sqrt(self.dim)
        embedding.weight.data.uniform_(-uniform_range, uniform_range)
        embedding.weight.data = torch.div(embedding.weight.data, embedding.weight.data.norm(p=2, dim=1, keepdim=True))
        return embedding

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
        positive_distances = self._distance(positive_triplets)
        negative_distances = self._distance(negative_triplets)
        return positive_distances, negative_distances

    def _distance(self, triplets):
        heads = self.entities_emb(triplets[:, 0])
        relations = self.relations_emb(triplets[:, 1])
        tails = self.entities_emb(triplets[:, 2])
        return (heads + relations - tails).norm(p=self.norm, dim=1)

    def link_predict(self, head, relation, tail=None, k=10):
        # h_add_r: [batch size, embed size] -> [batch size, 1, embed size] -> [batch size, entity num, embed size]
        h_add_r = self.entities_emb(head) + self.relations_emb(relation)
        h_add_r = torch.unsqueeze(h_add_r, dim=1)
        h_add_r = h_add_r.expand(h_add_r.shape[0], self.entity_num, self.dim)
        # embed_tail: [batch size, embed size] -> [batch size, entity num, embed size]
        embed_tail = self.entities_emb.weight.data.expand(h_add_r.shape[0], self.entity_num, self.dim)
        # values: [batch size, k] scores, the smaller, the better
        # indices: [batch size, k] indices of entities ranked by scores
        values, indices = (h_add_r - embed_tail).norm(p=self.norm, dim=2).topk(k, largest=False)

        return values, indices
    
