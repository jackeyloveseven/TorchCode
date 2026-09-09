"""OPD (On-Policy Distillation) Loss task."""

TASK = {
    "title": "OPD（在线策略蒸馏）损失",
    "difficulty": "Hard",
    "function_name": "opd_loss",
    "hint": (
        "计算从学生到每位教师的逆向 KL 散度："
        "KL(pi_student || pi_teacher) = sum_v p_student(v) * "
        "(log p_student(v) - log p_teacher(v))。用 teacher_weights 对各教师 KL 加权平均，"
        "若提供 mask 则在 token 维度上应用，最后乘以 temperature ** 2。"
    ),
    "tests": [
        {
            "name": "基本形状与类型",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "student_logits = torch.randn(2, 3, 5, requires_grad=True)\n"
            "teacher_logits = torch.randn(2, 3, 5)\n"
            "loss = {fn}(student_logits, teacher_logits)\n"
            "assert isinstance(loss, Tensor) and loss.dim() == 0, 'Loss must be a scalar Tensor'\n"
        },
        {
            "name": "学生与教师匹配时损失为零",
            "code": "\n"
            "import torch\n"
            "torch.manual_seed(0)\n"
            "student_logits = torch.randn(2, 3, 4, requires_grad=True)\n"
            "teacher_logits = student_logits.detach().clone()\n"
            "loss = {fn}(student_logits, teacher_logits)\n"
            "assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), f'Expected near-zero loss, got {loss.item():.8f}'\n"
        },
        {
            "name": "数值对比单教师逆向 KL",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "student_logits = torch.tensor([[[2.0, 0.0, -1.0], [0.5, 1.0, -0.5]]])\n"
            "teacher_logits = torch.tensor([[[1.0, 1.5, -0.5], [0.0, 2.0, -1.0]]])\n"
            "student_logits = student_logits.requires_grad_()\n"
            "loss = {fn}(student_logits, teacher_logits)\n"
            "s_logp = F.log_softmax(student_logits, dim=-1)\n"
            "t_logp = F.log_softmax(teacher_logits, dim=-1)\n"
            "expected = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1).mean()\n"
            "assert torch.allclose(loss, expected, atol=1e-6), f'{loss.item():.6f} vs {expected.item():.6f}'\n"
        },
        {
            "name": "多教师加权逆向 KL",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "student_logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)\n"
            "teacher_logits = torch.stack([\n"
            "    torch.tensor([[[2.0, -1.0], [1.0, 0.0]]]),\n"
            "    torch.tensor([[[-1.0, 2.0], [0.0, 1.0]]]),\n"
            "])\n"
            "weights = torch.tensor([0.25, 0.75])\n"
            "loss = {fn}(student_logits, teacher_logits, teacher_weights=weights)\n"
            "s_logp = F.log_softmax(student_logits, dim=-1)\n"
            "s_prob = s_logp.exp()\n"
            "t_logp = F.log_softmax(teacher_logits, dim=-1)\n"
            "kl = (s_prob.unsqueeze(0) * (s_logp.unsqueeze(0) - t_logp)).sum(dim=-1)\n"
            "expected = (weights.view(-1, 1, 1) * kl).sum(dim=0).mean()\n"
            "assert torch.allclose(loss, expected, atol=1e-6), f'{loss.item():.6f} vs {expected.item():.6f}'\n"
        },
        {
            "name": "掩码忽略填充 token",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "student_logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]], requires_grad=True)\n"
            "teacher_logits = torch.tensor([[[0.0, 2.0], [0.0, 2.0], [100.0, -100.0]]])\n"
            "mask = torch.tensor([[1.0, 1.0, 0.0]])\n"
            "loss = {fn}(student_logits, teacher_logits, mask=mask)\n"
            "s_logp = F.log_softmax(student_logits, dim=-1)\n"
            "t_logp = F.log_softmax(teacher_logits, dim=-1)\n"
            "per_token = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)\n"
            "expected = (per_token * mask).sum() / mask.sum()\n"
            "assert torch.allclose(loss, expected, atol=1e-6), 'Masked positions should not affect the loss'\n"
        },
        {
            "name": "梯度仅流向学生 logits",
            "code": "\n"
            "import torch\n"
            "student_logits = torch.randn(2, 3, 4, requires_grad=True)\n"
            "teacher_logits = torch.randn(2, 3, 4, requires_grad=True)\n"
            "loss = {fn}(student_logits, teacher_logits)\n"
            "loss.backward()\n"
            "assert student_logits.grad is not None, 'Student logits should receive gradients'\n"
            "assert teacher_logits.grad is None, 'Teacher logits should be treated as frozen targets'\n"
        },
        {
            "name": "温度缩放",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "student_logits = torch.tensor([[[2.0, 0.0, -1.0]]], requires_grad=True)\n"
            "teacher_logits = torch.tensor([[[0.0, 1.0, -1.0]]])\n"
            "temperature = 2.0\n"
            "loss = {fn}(student_logits, teacher_logits, temperature=temperature)\n"
            "s_logp = F.log_softmax(student_logits / temperature, dim=-1)\n"
            "t_logp = F.log_softmax(teacher_logits / temperature, dim=-1)\n"
            "expected = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1).mean() * (temperature ** 2)\n"
            "assert torch.allclose(loss, expected, atol=1e-6), f'{loss.item():.6f} vs {expected.item():.6f}'\n"
        },
    ],
}
