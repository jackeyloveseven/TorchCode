"""Softmax implementation task."""

TASK = {
    "title": "实现 Softmax",
    "difficulty": "Easy",
    "function_name": "my_softmax",
    "hint": "softmax(x)_i = exp(x_i) / sum(exp(x_j))。先减去 max(x) 以保证数值稳定性。",
    "tests": [
        {
            "name": "基本一维",
            "code": """
import torch
x = torch.tensor([1.0, 2.0, 3.0])
out = {fn}(x, dim=-1)
expected = torch.softmax(x, dim=-1)
assert torch.allclose(out, expected, atol=1e-5), f'{out} vs {expected}'
""",
        },
        {
            "name": "沿 dim=-1 的二维",
            "code": """
import torch
x = torch.randn(4, 8)
out = {fn}(x, dim=-1)
expected = torch.softmax(x, dim=-1)
assert out.shape == expected.shape, f'Shape mismatch'
assert torch.allclose(out, expected, atol=1e-5), 'Values differ'
assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5), 'Rows must sum to 1'
""",
        },
        {
            "name": "数值稳定性",
            "code": """
import torch
x = torch.tensor([1000., 1001., 1002.])
out = {fn}(x, dim=-1)
assert not torch.isnan(out).any(), 'NaN in output — not numerically stable'
assert not torch.isinf(out).any(), 'Inf in output — not numerically stable'
expected = torch.softmax(x, dim=-1)
assert torch.allclose(out, expected, atol=1e-5), 'Values differ on large input'
""",
        },
    ],
}
