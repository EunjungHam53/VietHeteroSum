import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, layers=2, act=nn.LeakyReLU(), dropout_p=0.3, keep_last_layer=False):
        super(MLP, self).__init__()
        self.layers = layers
        self.act = act
        self.dropout = nn.Dropout(dropout_p)
        self.keep_last = keep_last_layer

        self.mlp_layers = nn.ModuleList([])
        if layers == 1:
            self.mlp_layers.append(nn.Linear(in_dim, out_dim))
        else:
            self.mlp_layers.append(nn.Linear(in_dim, hid_dim))
            for i in range(self.layers - 2):
                self.mlp_layers.append(nn.Linear(hid_dim, hid_dim))
            self.mlp_layers.append(nn.Linear(hid_dim, out_dim))

    def forward(self, x):
        for i in range(len(self.mlp_layers) - 1):
            x = self.dropout(self.act(self.mlp_layers[i](x)))
        if self.keep_last:
            x = self.mlp_layers[-1](x)
        else:
            x = self.act(self.mlp_layers[-1](x))
        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True, dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        n_nodes = h.shape[0]
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_repeat = g.repeat(n_nodes, 1, 1)
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn(g_concat)).squeeze(-1)

        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        e = e.masked_fill(adj_mat == 0, float(-1e4))
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('ijh,jhf->ihf', a, g)

        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)

class GAT(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):
        super().__init__()
        self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout)
        self.activation = nn.ELU()
        self.output = GraphAttentionLayer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        x = x.squeeze(0)
        adj_mat = adj_mat.squeeze(0).unsqueeze(-1).bool()

        x = self.dropout(x)
        x = self.layer1(x, adj_mat)

        x = self.activation(x)
        x = self.dropout(x)
        return self.output(x, adj_mat).unsqueeze(0)

class NodeScorer(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
    def forward(self, x):
        return torch.tanh(self.linear(x))
class Gate(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.scorer = NodeScorer(in_features)
    def forward(self, x):
        scores = self.scorer(x).squeeze(-1)
        x = x * scores.view(-1,1)
        return x

class StepWiseGraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, dropout_p=0.3, act=nn.LeakyReLU(), nheads=6, iter=1, have_gate=False):
        super().__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout_p)
        self.iter = iter
        self.edu_gat = nn.ModuleList([GAT(in_features=in_dim, n_hidden=hid_dim, n_classes=in_dim,
                                          dropout=dropout_p, n_heads=nheads) for _ in range(iter)])
        self.sent_gat = nn.ModuleList([GAT(in_features=in_dim, n_hidden=hid_dim, n_classes=in_dim,
                                           dropout=dropout_p, n_heads=nheads) for _ in range(iter)])
        self.sec_gat = nn.ModuleList([GAT(in_features=in_dim, n_hidden=hid_dim, n_classes=in_dim,
                                          dropout=dropout_p, n_heads=nheads) for _ in range(iter)])

        self.feature_fusion_layer = nn.Linear(in_dim * 3, in_dim)
        self.ffn = MLP(in_dim, in_dim, hid_dim, dropout_p=dropout_p, layers=3)
        self.have_gate = have_gate
        if self.have_gate:
            self.gate = Gate(in_dim)

    def forward(self, feature, adj, sect_num, sent_num):
        edu_adj = adj.clone()
        edu_adj[:, :, -sect_num - sent_num:] = 0
        sent_adj = adj.clone()
        sent_adj[:, :, :-sect_num - sent_num] = sent_adj[:, :, -sect_num:] = 0
        sect_adj = adj.clone()
        sect_adj[:, :, :-sect_num] = 0

        feature_edu = feature.clone()
        feature_sent = feature.clone()
        feature_sect = feature.clone()
        feature_resi = feature

        feature_edu_re = feature_edu
        feature_sent_re = feature_sent
        feature_sect_re = feature_sect

        for i in range(0, self.iter):
            feature_edu = self.edu_gat[i](feature_edu, edu_adj)
        feature_edu += feature_edu_re

        for i in range(0, self.iter):
            feature_sent = self.sent_gat[i](feature_sent, sent_adj)
        feature_sent += feature_sent_re

        for i in range(0, self.iter):
            feature_sect = self.sec_gat[i](feature_sect, sect_adj)
        feature_sect += feature_sect_re

        feature = torch.concat([feature_sect, feature_sent, feature_edu], dim=-1)
        feature = self.dropout(F.leaky_relu(self.feature_fusion_layer(feature)))
        feature = self.ffn(feature) + feature_resi

        if self.have_gate:
            feature = self.gate(feature)
            
        return feature

class Contrast_Encoder(nn.Module):
    def __init__(self, graph_encoder, hidden_dim, in_dim=768, dropout_p=0.3):
        super(Contrast_Encoder, self).__init__()
        self.graph_encoder = graph_encoder

    def forward(self, p_gfeature, p_adj, sect_num, sent_num):
        pg = self.graph_encoder(p_gfeature.float(), p_adj.float(), sect_num, sent_num)
        return pg

class End2End_Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout_p):
        super(End2End_Encoder, self).__init__()
        self.dropout = nn.Dropout(dropout_p)

        sec_dim =  int(in_dim / 4)
        sent_dim =  int(in_dim / 2)
        mlp_deep = 5
        out_dim = int(in_dim/2)

        self.linear_sec = nn.Linear(in_dim, sec_dim)
        self.linear_sent = nn.Linear(in_dim, sent_dim)
        self.layer_norm = nn.LayerNorm(normalized_shape=in_dim + sent_dim + sec_dim)
        self.out_proj_layer_mlp = MLP(in_dim + sent_dim + sec_dim, out_dim, hidden_dim, act=nn.LeakyReLU(), dropout_p=dropout_p, layers= mlp_deep)
        self.final_layer = nn.Linear(out_dim, 1)

    def forward(self, x, adj, sect_num, sent_num):
        batch_size, num_nodes, feat_dim = x.shape
        edu_part = x[:, :-sent_num - sect_num, :]
        sent_part = x[:, -sent_num - sect_num:-sect_num, :]
        sect_part = x[:, -sect_num:, :]

        sent_indices = []
        sect_indices = []

        for i in range(batch_size):
            sent_mask = adj[i, :-sent_num - sect_num, -sent_num - sect_num:-sect_num] == 1
            sect_mask = (adj[i, :-sent_num - sect_num, -sent_num - sect_num:-sect_num] @ adj[i, -sent_num - sect_num:-sect_num, -sect_num:]) == 1

            sent_idx = sent_mask.nonzero(as_tuple=True)[1]
            sect_idx = sect_mask.nonzero(as_tuple=True)[1]

            if len(sent_idx) == 0:
                print(
                    f"sent_idx is empty, sent_mask: {sent_mask.cpu().numpy()}\nAdj slice: {adj[i, :-sent_num - sect_num, -sent_num - sect_num:-sect_num].cpu().numpy()}\nShape of edu_part: {edu_part.shape}, Device: {x.device}, EDU indices: {list(range(edu_part.shape[1]))}")
                sent_idx = torch.zeros(edu_part.shape[1], dtype=torch.long, device=x.device)

            if len(sect_idx) == 0:
                print(
                    f"sect_idx is empty, sect_mask: {sect_mask.cpu().numpy()}\nAdj slice: {adj[i, :-sent_num - sect_num, -sect_num:].cpu().numpy()}\nShape of edu_part: {edu_part.shape}, Device: {x.device}, EDU indices: {list(range(edu_part.shape[1]))}")
                sect_idx = torch.zeros(edu_part.shape[1], dtype=torch.long, device=x.device)

            sent_indices.append(sent_idx)
            sect_indices.append(sect_idx)

        sent_indices = torch.stack(sent_indices)
        sect_indices = torch.stack(sect_indices)

        edu_to_sent = torch.gather(sent_part, 1, sent_indices.unsqueeze(-1).expand(-1, -1, feat_dim))
        edu_to_sect = torch.gather(sect_part, 1, sect_indices.unsqueeze(-1).expand(-1, -1, feat_dim))

        x_combined = torch.cat([edu_part, self.linear_sent(edu_to_sent), self.linear_sec(edu_to_sect)], dim=-1)
        x_combined = self.layer_norm(x_combined)
        x = self.out_proj_layer_mlp(x_combined)
        x = self.final_layer(x)
        return x

class End2End_Encoder_EDU(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout_p, mlp_layers=2):
        super(End2End_Encoder_EDU, self).__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.out_proj_layer_mlp = MLP(in_dim, in_dim, hidden_dim, act=nn.LeakyReLU(), dropout_p=dropout_p, layers=mlp_layers)
        self.final_layer = nn.Linear(in_dim, 1)

    def forward(self, x, adj, sect_num, sent_num):
        x = x[:, :-sect_num - sent_num, :]
        x = self.out_proj_layer_mlp(x)
        x = self.final_layer(x)
        return x