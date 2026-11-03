import torch
import torch.nn.functional as F

def manuscript_prior_preservation(student, teacher, model):
    fused = student['bag_feat']
    names = student['ordered_names']
    if names != teacher['ordered_names']:
        raise ValueError('Student and teacher branch orders differ.')
    branch_losses = []
    for index, name in enumerate(names):
        heads = (model.original_prior_heads[name], model.projected_prior_heads[name], model.remapped_prior_heads[name])
        targets = (teacher['raw_bag_feat'][name], teacher['proj_bag_feat'][:, index], teacher['remap_bag_feat'][:, index])
        stage_losses = [1 - F.cosine_similarity(head(fused), target.detach(), dim=-1) for head, target in zip(heads, targets)]
        branch_losses.append(torch.stack(stage_losses, dim=-1).sum(dim=-1))
    prior = (torch.stack(branch_losses, dim=-1) * student['weights']).sum(dim=-1).mean()
    consistency = (1 - F.cosine_similarity(fused, teacher['bag_feat'].detach(), dim=-1)).mean()
    return prior, consistency

def routing_target(aux_logits, labels, temperature):
    if temperature <= 0:
        raise ValueError('Routing temperature must be positive.')
    batch, branches, classes = aux_logits.shape
    probabilities = aux_logits.detach().softmax(dim=-1)
    evidence = probabilities.gather(2, labels.view(batch, 1, 1).expand(batch, branches, 1)).squeeze(-1)
    return (evidence / temperature).softmax(dim=-1)

def compute_objective(student, teacher, labels, model, config):
    logits = student['logits']
    if config.get('use_logit_norm', False):
        temperature = config.get('logit_norm_temperature', 0.1)
        if temperature <= 0:
            raise ValueError('Logit normalization temperature must be positive.')
        logits = F.normalize(logits, p=2, dim=-1) / temperature
    main = F.cross_entropy(logits, labels)
    aux = student['aux_logits']
    batch, branches, classes = aux.shape
    expanded = labels.view(batch, 1).expand(batch, branches).reshape(-1)
    auxiliary = 0.3 * F.cross_entropy(aux.reshape(-1, classes), expanded, reduction='none').view(batch, branches).sum(dim=1).mean()
    target = routing_target(aux, labels, config['routing_temperature'])
    routing = F.kl_div(torch.log(student['weights'] + 1e-8), target, reduction='batchmean')
    if config['prior_implementation'] != 'manuscript_reference':
        raise ValueError('Only the explicitly identified manuscript_reference prior is supplied. The historical loss source was not available.')
    if config['lambda_reg_effective'] > 0:
        prior, consistency = manuscript_prior_preservation(student, teacher, model)
    else:
        prior = consistency = main.new_zeros(())
    regularization = prior + consistency
    coefficients = [config.get('lambda_main', 1.0), config['lambda_aux'], config['lambda_route'], config['lambda_reg_effective']]
    if any(not torch.isfinite(torch.tensor(value)) or value < 0 for value in coefficients):
        raise ValueError('Objective coefficients must be finite and nonnegative.')
    total = config.get('lambda_main', 1.0) * main + config['lambda_aux'] * auxiliary + config['lambda_route'] * routing + config['lambda_reg_effective'] * regularization
    metrics = {'main': main, 'auxiliary': auxiliary, 'routing': routing, 'prior': prior, 'consistency': consistency}
    return total, {key: float(value.detach()) for key, value in metrics.items()}
