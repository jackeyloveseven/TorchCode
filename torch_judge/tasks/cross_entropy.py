"""Cross-Entropy Loss task."""

TASK = {
    "title": "交叉熵损失",
    "difficulty": "Easy",
    "function_name": "cross_entropy_loss",
    "hint": "log_probs = logits - logsumexp(logits, dim=-1, keepdim=True)。Loss = -log_probs[arange(B), targets].mean()。减去最大值以保证数值稳定（logsumexp 已处理此问题）。",
    "tests": [
        {
            "name": "与 F.cross_entropy 结果一致",
            "code": "\nimport torch\ntorch.manual_seed(0)\nlogits = torch.randn(4, 10)\ntargets = torch.randint(0, 10, (4,))\nout = {fn}(logits, targets)\nref = torch.nn.functional.cross_entropy(logits, targets)\nassert torch.allclose(out, ref, atol=1e-5), f'Mismatch: {out.item():.4f} vs {ref.item():.4f}'\n"
        },
        {
            "name": "数值稳定性",
            "code": "\nimport torch\nlogits = torch.tensor([[1000., 0., 0.], [0., 1000., 0.]])\ntargets = torch.tensor([0, 1])\nout = {fn}(logits, targets)\nassert not torch.isnan(out), 'NaN with large logits'\nassert not torch.isinf(out), 'Inf with large logits'\nassert out.item() < 0.01, 'Should be ~0 for confident correct predictions'\n"
        },
        {
            "name": "标量输出",
            "code": "\nimport torch\nout = {fn}(torch.randn(8, 5), torch.randint(0, 5, (8,)))\nassert out.dim() == 0, 'Loss must be a scalar'\n"
        },
        {
            "name": "梯度流",
            "code": "\nimport torch\nlogits = torch.randn(8, 5, requires_grad=True)\ntargets = torch.randint(0, 5, (8,))\n{fn}(logits, targets).backward()\nassert logits.grad is not None, 'logits.grad is None'\n"
        }
    ]
}
