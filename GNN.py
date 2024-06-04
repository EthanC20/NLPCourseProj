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


if __name__ == '__main__':
    # 读取数据
    with open('OpenBG500/OpenBG500_entity2text.tsv', 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        lines = [line.strip('\n').split('\t') for line in dat]
    ent2id = {line[0]: i for i, line in enumerate(lines)}
    with open('OpenBG500/OpenBG500_relation2text.tsv', 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        lines = [line.strip().split('\t') for line in dat]
    rel2id = {line[0]: i for i, line in enumerate(lines)}
    with open('OpenBG500/OpenBG500_train.tsv', 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        train = [line.strip('\n').split('\t') for line in dat]
    with open('OpenBG500/OpenBG500_dev.tsv', 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        dev = [line.strip('\n').split('\t') for line in dat]

    train_dataset = TripleDataset(ent2id, rel2id, train)
    dev_dataset = TripleDataset(ent2id, rel2id, dev)
    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = data.DataLoader(dev_dataset, batch_size=32, shuffle=False)

    model = GNN(len(ent2id), len(rel2id))
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MarginRankingLoss(margin=1.0)

    print("start training")
    for epoch in range(10):
        model.train()
        for i, (positive_triplets, negative_triplets) in enumerate(tqdm.tqdm(train_loader)):
            positive_triplets = positive_triplets.cuda()
            negative_triplets = negative_triplets.cuda()
            positive_distances, negative_distances = model(positive_triplets, negative_triplets)
            target = torch.ones(positive_distances.shape[0]).cuda()
            loss = criterion(positive_distances, negative_distances, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        mrr, hits_1, hits_3, hits_10 = model.evaluate(dev_loader)
        print(f"Epoch {epoch}: MRR: {mrr}, Hits@1: {hits_1}, Hits@3: {hits_3}, Hits@10: {hits_10}")