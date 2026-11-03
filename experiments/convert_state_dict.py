import argparse
import torch


KEY_MAPPING = {'feature_emb': 'shared_remapping', 'attention': 'mil_pooling', 'weight_assign': 'dynamic_routing', 'cross_attn': 'cross_model_attention', 'cross_ffn': 'fusion_ffn', 'proj_distill_heads': 'projected_prior_heads', 'remap_distill_heads': 'remapped_prior_heads', 'fused_distill_heads': 'original_prior_heads'}


def convert(state):
    converted = {'.'.join(KEY_MAPPING.get(part, part) for part in key.split('.')): value for key, value in state.items()}
    if len(converted) != len(state):
        raise ValueError('Renaming produced duplicate parameter keys.')
    return converted


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    state = torch.load(args.input, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError('Input must be a tensor-only model state dictionary.')
    torch.save(convert(state), args.output)
