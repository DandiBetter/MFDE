import torch
from torch import nn
import torch.nn.functional as F
from .base import BaseMIL
from .components import DistillationHead, MultiHeadMILAttention, PrototypeAssistedSharedRemapping, WeightAssignmentOperator

class MFDE(BaseMIL):

    def __init__(self, instance_dim, embed_instance_dim, bag_dim, n_classes, num_attention_heads, attention_dim, gated, bias, dropout, act, num_prototypes=32):
        super().__init__(instance_dim, embed_instance_dim, bag_dim, n_classes)
        self.d_attn_kwargs = {'embed_instance_dim': embed_instance_dim, 'bag_dim': bag_dim, 'n_classes': n_classes, 'num_attention_heads': num_attention_heads, 'attention_dim': attention_dim, 'gated': gated, 'bias': bias, 'dropout': dropout, 'act': act}
        self.instance_dim = instance_dim
        self.embed_instance_dim = embed_instance_dim
        self.num_attention_heads = num_attention_heads
        self.n_classes = n_classes
        self.projectors = nn.ModuleDict()
        self.classifier = None
        self.shared_remapping = None
        self.mil_pooling = None
        self.head_gate = None
        self.cross_model_attention = None
        self.fusion_ffn = None
        self.model_embeddings = None
        self.sample_match_proj = None
        self.model_match_proj = None
        self.match_score = None
        self.route_feat_proj = None
        self.model_names = None
        self.num_models = None
        self.num_prototypes = num_prototypes
        self.original_prior_heads = nn.ModuleDict()
        self.projected_prior_heads = nn.ModuleDict()
        self.remapped_prior_heads = nn.ModuleDict()
        self.raw_distill_heads = nn.ModuleDict()
        self.distill_dim = max(64, embed_instance_dim // 2)

    def _compute_attn_stats(self, attn: torch.Tensor, topk: int=5, eps: float=1e-08):
        B, M, N = attn.shape
        topk = min(topk, N)
        entropy = -(attn * torch.log(attn + eps)).sum(dim=-1, keepdim=True)
        mean = attn.mean(dim=-1, keepdim=True)
        std = torch.sqrt(((attn - mean) ** 2).mean(dim=-1, keepdim=True) + eps)
        amax = attn.max(dim=-1, keepdim=True).values
        topk_mass = torch.topk(attn, k=topk, dim=-1).values.sum(dim=-1, keepdim=True)
        stats = torch.cat([entropy, std, amax, topk_mass], dim=-1)
        return stats

    def get_submodules(self, feat_dict: dict):
        self.model_names = list(feat_dict.keys())
        self.num_models = len(self.model_names)
        for name, feat in feat_dict.items():
            in_dim = feat.shape[-1]
            self.projectors[name] = nn.Sequential(nn.Linear(in_dim, self.embed_instance_dim, bias=False), nn.LayerNorm(self.embed_instance_dim))
            self.projected_prior_heads[name] = DistillationHead(in_features=self.embed_instance_dim, out_features=self.embed_instance_dim, act_layer=nn.GELU, norm_layer=nn.Identity)
            self.remapped_prior_heads[name] = DistillationHead(in_features=self.embed_instance_dim, out_features=self.embed_instance_dim, act_layer=nn.GELU, norm_layer=nn.Identity)
            self.original_prior_heads[name] = DistillationHead(in_features=self.embed_instance_dim, out_features=in_dim, act_layer=nn.GELU, norm_layer=nn.Identity)
        self.shared_remapping = PrototypeAssistedSharedRemapping(d_model=self.embed_instance_dim, num_proxies=self.num_prototypes, n_heads=4, mlp_ratio=2.0)
        self.mil_pooling = MultiHeadMILAttention(input_dim=self.embed_instance_dim, attention_dim=self.d_attn_kwargs['attention_dim'], num_attention_heads=self.d_attn_kwargs['num_attention_heads'], act='relu', bias=self.d_attn_kwargs['bias'], dropout=self.d_attn_kwargs['dropout'])
        self.head_gate = nn.Linear(self.embed_instance_dim, 1)
        self.cross_model_attention = nn.MultiheadAttention(self.embed_instance_dim, 4, batch_first=True)
        self.fusion_ffn = nn.Sequential(nn.LayerNorm(self.embed_instance_dim), nn.Linear(self.embed_instance_dim, self.embed_instance_dim), nn.GELU(), nn.Linear(self.embed_instance_dim, self.embed_instance_dim))
        self.classifier = nn.Linear(self.embed_instance_dim, self.n_classes)
        self.dynamic_routing = WeightAssignmentOperator(embed_dim=self.embed_instance_dim, num_models=self.num_models, stats_dim=4)

    def forward(self, feat_dict: dict, return_attn=True, return_bag_feat=True):
        if len(self.projectors) == 0:
            self.get_submodules(feat_dict)
        x_list = []
        x_projected_list = []
        ordered_names = ['uni'] + [n for n in self.model_names if n != 'uni'] if 'uni' in self.model_names else self.model_names
        for name in ordered_names:
            x = feat_dict[name]
            x_list.append(x)
            x_projected = self.projectors[name](x)
            x_projected_list.append(x_projected)
        M = len(x_projected_list)
        B, N, D = x_projected_list[0].shape
        x_projected_stack = torch.stack(x_projected_list, dim=1)
        x_projected_flat = x_projected_stack.view(B * M, N, D)
        h, proxies = self.shared_remapping(x_projected_flat)
        A = self.mil_pooling(h)
        A_heads = F.softmax(A, dim=-1)
        bag_feats_heads = torch.einsum('bhn,bnd->bhd', A_heads, h)
        head_score = self.head_gate(bag_feats_heads).squeeze(-1)
        head_w = F.softmax(head_score, dim=-1)
        A_softmax = (A_heads * head_w.unsqueeze(-1)).sum(dim=1)
        bag_feat = torch.einsum('bn,bnd->bd', A_softmax, h)
        context = bag_feat.view(B, M, D)
        proj_bag_feat = torch.einsum('bn,bnd->bd', A_softmax, x_projected_flat).view(B, M, D)
        raw_bag_dict = {}
        for m, name in enumerate(ordered_names):
            a_m = A_softmax.view(B, M, N)[:, m, :]
            x_m = x_list[m]
            raw_bag_dict[name] = torch.einsum('bn,bnd->bd', a_m, x_m)
        attn_single = A_softmax.view(B, M, N)
        attn_stats = self._compute_attn_stats(attn_single, topk=5)
        weights, scores, route_feat = self.dynamic_routing(context, attn_stats)
        z0 = (context * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
        attn_out, cross_attn_weights = self.cross_model_attention(z0, context, context)
        fused_feat = (z0 + attn_out).squeeze(1)
        fused_feat = fused_feat + self.fusion_ffn(fused_feat)
        mul_logits = self.classifier(context)
        logits = self.classifier(fused_feat)
        ret = {'logits': logits, 'aux_logits': mul_logits, 'weights': weights, 'route_scores': scores}
        if return_bag_feat:
            ret['bag_feat'] = fused_feat
            ret['aux_feat'] = context
            ret['route_feat'] = route_feat
            ret['attn_stats'] = attn_stats
            ret['proj_bag_feat'] = proj_bag_feat
            ret['remap_bag_feat'] = context
            ret['raw_bag_feat'] = raw_bag_dict
            ret['ordered_names'] = ordered_names
            ret['proxies'] = proxies
        if return_attn:
            ret['attn'] = A_softmax
            ret['attn_ori'] = A
            ret['head_weights'] = head_w.view(B, M, -1)
            ret['cross_attn_weights'] = cross_attn_weights
        return ret
