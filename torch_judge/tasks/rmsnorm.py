"""RMSNorm implementation task."""

TASK = {
    "title": "实现 RMSNorm",
    "difficulty": "Medium",
    "function_name": "rms_norm",
    "hint": "$\\text{RMS}(x) = \\sqrt{\\text{mean}(x^2) + \\epsilon}$。$\\text{RMSNorm}(x) = \\frac{x}{\\text{RMS}(x)} \\cdot \\text{weight}$。比 LayerNorm 更简单——无需减去均值。",
    "tests": [
        {
            "name": "基本行为",
            "code": """
import torch
x = torch.randn(2, 8)
weight = torch.ones(8)
out = {fn}(x, weight)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
ref = x / rms * weight
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch'
""",
        },
        {
            "name": "带可学习权重",
            "code": """
import torch
x = torch.randn(4, 16)
weight = torch.randn(16)
out = {fn}(x, weight)
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
ref = x / rms * weight
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch with non-trivial weight'
""",
        },
        {
            "name": "三维输入",
            "code": """
import torch
x = torch.randn(2, 4, 32)
weight = torch.ones(32)
out = {fn}(x, weight)
assert out.shape == x.shape, f'Shape mismatch on 3-D: {out.shape}'
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
ref = x / rms * weight
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch on 3-D'
""",
        },
        {
            "name": "梯度流",
            "code": """
import torch
x = torch.randn(2, 8, requires_grad=True)
weight = torch.ones(8, requires_grad=True)
out = {fn}(x, weight)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert weight.grad is not None, 'weight.grad is None'
""",
        },
    ],
}
