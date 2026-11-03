import torch
from torch import nn
import torch.nn.functional as F

class DistillationHead(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, norm_layer=nn.LayerNorm, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.norm1 = norm_layer(in_features) if norm_layer is not None else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

    def forward_feature(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class MultiHeadMILAttention(nn.Module):

    def __init__(self, input_dim=512, attention_dim=128, num_attention_heads=1, act='relu', bias=False, dropout=False):
        super(MultiHeadMILAttention, self).__init__()
        self.L = input_dim
        self.D = attention_dim
        self.K = num_attention_heads
        self.mil_pooling = [nn.Linear(self.L, self.D, bias=bias)]
        if act == 'gelu':
            self.mil_pooling += [nn.GELU()]
        elif act == 'relu':
            self.mil_pooling += [nn.ReLU()]
        elif act == 'tanh':
            self.mil_pooling += [nn.Tanh()]
        if dropout:
            self.mil_pooling += [nn.Dropout(0.25)]
        self.mil_pooling += [nn.Linear(self.D, self.K, bias=bias)]
        self.mil_pooling = nn.Sequential(*self.mil_pooling)

    def forward(self, x):
        A = self.mil_pooling(x)
        A = torch.transpose(A, -1, -2)
        return A

class PrototypeAssistedSharedRemapping(nn.Module):

    def __init__(self, d_model=512, num_proxies=64, n_heads=8, mlp_ratio=2.0):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(1, num_proxies, d_model) * 0.02)
        self.proxy_condense_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.proxy_broadcast_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_x_in = nn.Identity()
        self.norm_proxy = nn.LayerNorm(d_model)
        self.norm_x_ffn = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(0.0), nn.Linear(hidden_dim, d_model), nn.Dropout(0.0))

    def forward(self, x):
        B = x.shape[0]
        proxies = self.proxies.expand(B, -1, -1)
        normed_x = self.norm_x_in(x)
        condensed_proxies, _ = self.proxy_condense_attn(query=proxies, key=normed_x, value=normed_x)
        proxies = proxies + condensed_proxies
        normed_proxies = self.norm_proxy(proxies)
        updated_x, _ = self.proxy_broadcast_attn(query=x, key=normed_proxies, value=normed_proxies)
        x = x + updated_x
        out = x + self.mlp(self.norm_x_ffn(x))
        return (out, proxies)

class WeightAssignmentOperator(nn.Module):

    def __init__(self, embed_dim: int, num_models: int, stats_dim: int=4, match_dim: int | None=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_models = num_models
        self.stats_dim = stats_dim
        self.match_dim = match_dim or max(32, embed_dim // 2)
        self.route_feat_proj = nn.Sequential(nn.Linear(embed_dim + stats_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.model_embeddings = nn.Parameter(torch.randn(num_models, embed_dim) * 0.02)
        self.sample_match_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, self.match_dim))
        self.model_match_proj = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Linear(embed_dim, self.match_dim))
        self.match_score = nn.Sequential(nn.Linear(self.match_dim, self.match_dim), nn.GELU(), nn.Linear(self.match_dim, 1))

    def forward(self, context: torch.Tensor, attn_stats: torch.Tensor):
        B, M, D = context.shape
        assert M == self.num_models, f'Expected {self.num_models} models, got {M}'
        assert attn_stats.shape[:2] == (B, M)
        attn_stats_norm = F.layer_norm(attn_stats, (attn_stats.size(-1),))
        route_input = torch.cat([context, attn_stats_norm], dim=-1)
        route_feat = self.route_feat_proj(route_input)
        r_bar = route_feat.mean(dim=1)
        c_x = self.sample_match_proj(r_bar)
        model_embed = self.model_embeddings.unsqueeze(0).expand(B, -1, -1)
        route_feat_norm = F.layer_norm(route_feat, (route_feat.size(-1),))
        model_input = torch.cat([model_embed, route_feat_norm], dim=-1)
        c_m = self.model_match_proj(model_input)
        c_x_expand = c_x.unsqueeze(1).expand(-1, M, -1)
        u_m = c_x_expand * c_m
        scores = self.match_score(u_m).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        return (weights, scores, route_feat)
