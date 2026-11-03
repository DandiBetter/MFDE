import copy
import unittest
import torch
from mfde import PriorPreservationTeacher
from mfde.objective import compute_objective, routing_target
from train import build_model


class MFDETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.features = {'uni': torch.randn(1, 9, 12), 'gigapath': torch.randn(1, 9, 20), 'phikon': torch.randn(1, 9, 16)}
        self.config = {'embed_dim': 16, 'num_classes': 2, 'attention_heads': 4, 'attention_dim': 8, 'attention_dropout': 0.25, 'num_prototypes': 8}
        self.model = build_model(self.config, self.features)
        self.objective = {'prior_implementation': 'manuscript_reference', 'routing_temperature': 0.1, 'lambda_aux': 0.5, 'lambda_route': 1, 'lambda_reg_effective': 1}

    def test_forward_and_gradient(self):
        teacher = PriorPreservationTeacher(self.model)
        output = self.model(self.features)
        loss, _ = compute_objective(output, teacher(self.features), torch.tensor([1]), self.model, self.objective)
        loss.backward()
        self.assertEqual(tuple(output['logits'].shape), (1, 2))
        torch.testing.assert_close(output['weights'].sum(-1), torch.ones(1))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(self.model.shared_remapping.proxies.grad)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_teacher_update(self):
        teacher = PriorPreservationTeacher(self.model, decay=0.5)
        initial = copy.deepcopy(teacher.teacher.state_dict())
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.add_(1)
        teacher.update(self.model)
        for name, value in teacher.teacher.state_dict().items():
            torch.testing.assert_close(value, (initial[name] + self.model.state_dict()[name]) * 0.5)
        self.assertFalse(teacher.teacher.training)

    def test_temperature_and_effective_scaling(self):
        aux = torch.tensor([[[2.0, -1.0], [0.0, 1.0], [1.0, 0.0]]], requires_grad=True)
        cold = routing_target(aux, torch.tensor([0]), 0.025)
        warm = routing_target(aux, torch.tensor([0]), 0.4)
        self.assertGreater(float(cold.max()), float(warm.max()))
        self.assertFalse(cold.requires_grad)
        self.model.eval()
        teacher = PriorPreservationTeacher(self.model)
        output, targets = self.model(self.features), teacher(self.features)
        losses = []
        for weight in (0, 1, 2):
            config = {**self.objective, 'lambda_reg_effective': weight}
            losses.append(compute_objective(output, targets, torch.tensor([1]), self.model, config)[0])
        torch.testing.assert_close(losses[2] - losses[0], 2 * (losses[1] - losses[0]))


if __name__ == '__main__':
    unittest.main()
