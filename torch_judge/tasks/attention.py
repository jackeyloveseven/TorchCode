"""Softmax Attention task."""

TASK = {
    "title": "Softmax 注意力",
    "difficulty": "Hard",
    "function_name": "scaled_dot_product_attention",
    "hint": "$\\text{scores} = (Q K^T) / \\sqrt{d_k}$，再计算 $\\text{softmax}(\\text{scores}, \\text{dim}=-1) V$。使用 `torch.bmm` 进行批量矩阵乘法。",
    "tests": [
        {
            "name": "输出形状",
            "code": """
import torch, math
torch.manual_seed(42)
B, S, D = 2, 4, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V)
assert out.shape == (B, S, D), f'Shape mismatch: {out.shape} vs {(B, S, D)}'
""",
        },
        {
            "name": "数值正确性",
            "code": """
import torch, math
torch.manual_seed(42)
B, S, D = 2, 4, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)
weights = torch.softmax(scores, dim=-1)
ref = torch.bmm(weights, V)
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch vs reference'
""",
        },
        {
            "name": "梯度检验",
            "code": """
import torch, math
torch.manual_seed(42)
Q = torch.randn(2, 4, 8, requires_grad=True)
K = torch.randn(2, 4, 8, requires_grad=True)
V = torch.randn(2, 4, 8, requires_grad=True)
out = {fn}(Q, K, V)
out.sum().backward()
assert Q.grad is not None, 'Q.grad is None'
assert K.grad is not None, 'K.grad is None'
assert V.grad is not None, 'V.grad is None'
""",
        },
        {
            "name": "交叉注意力（seq_q != seq_k）",
            "code": """
import torch
Q = torch.randn(1, 3, 16)
K = torch.randn(1, 5, 16)
V = torch.randn(1, 5, 32)
out = {fn}(Q, K, V)
assert out.shape == (1, 3, 32), f'Cross-attention shape: {out.shape}'
""",
        },
    ],
}
