"""Multi-Head Attention task."""

TASK = {
    "title": "多头注意力",
    "difficulty": "Hard",
    "function_name": "MultiHeadAttention",
    "hint": (
        "使用 nn.Linear 做 Q/K/V/O 投影。d_k = d_model // num_heads。"
        "重塑为 (B, heads, S, d_k)，逐头计算 SDPA，拼接后做输出投影。"
    ),
    "tests": [
        {
            "name": "输出形状",
            "code": """
import torch
torch.manual_seed(0)
B, S, D, H = 2, 6, 32, 4
mha = {fn}(d_model=D, num_heads=H)
x = torch.randn(B, S, D)
out = mha.forward(x, x, x)
assert out.shape == (B, S, D), f'Shape mismatch: {out.shape} vs {(B, S, D)}'
""",
        },
        {
            "name": "使用形状正确的 nn.Linear",
            "code": """
import torch, torch.nn as nn
mha = {fn}(d_model=32, num_heads=4)
for name in ['W_q', 'W_k', 'W_v', 'W_o']:
    layer = getattr(mha, name)
    assert isinstance(layer, nn.Linear), f'{name} should be nn.Linear, got {type(layer)}'
    assert layer.weight.shape == (32, 32), f'{name}.weight shape: {layer.weight.shape}'
    assert layer.weight.requires_grad, f'{name}.weight must require grad'
""",
        },
        {
            "name": "数值与参考实现一致",
            "code": """
import torch, torch.nn as nn, math
torch.manual_seed(0)
D, H = 16, 2
d_k = D // H
mha = {fn}(d_model=D, num_heads=H)
Q = torch.randn(1, 4, D)
K = torch.randn(1, 4, D)
V = torch.randn(1, 4, D)
out = mha.forward(Q, K, V)
q = mha.W_q(Q).view(1, 4, H, d_k).transpose(1, 2)
k = mha.W_k(K).view(1, 4, H, d_k).transpose(1, 2)
v = mha.W_v(V).view(1, 4, H, d_k).transpose(1, 2)
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
attn = torch.matmul(weights, v)
ref = mha.W_o(attn.transpose(1, 2).contiguous().view(1, 4, D))
assert torch.allclose(out, ref, atol=1e-5), 'Output does not match reference'
""",
        },
        {
            "name": "梯度流",
            "code": """
import torch
torch.manual_seed(0)
mha = {fn}(d_model=16, num_heads=2)
x = torch.randn(1, 4, 16, requires_grad=True)
out = mha.forward(x, x, x)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert mha.W_q.weight.grad is not None, 'W_q.weight.grad is None'
assert mha.W_o.weight.grad is not None, 'W_o.weight.grad is None'
""",
        },
        {
            "name": "交叉注意力（seq_q != seq_k）",
            "code": """
import torch
mha = {fn}(d_model=32, num_heads=4)
Q = torch.randn(1, 3, 32)
K = torch.randn(1, 7, 32)
V = torch.randn(1, 7, 32)
out = mha.forward(Q, K, V)
assert out.shape == (1, 3, 32), f'Cross-attention shape: {out.shape}'
""",
        },
        {
            "name": "不同注意力头输出不同",
            "code": """
import torch
torch.manual_seed(42)
D, H = 16, 4
d_k = D // H
mha = {fn}(d_model=D, num_heads=H)
x = torch.randn(1, 4, D)
q = mha.W_q(x).view(1, 4, H, d_k).transpose(1, 2)
assert not torch.allclose(q[:, 0], q[:, 1], atol=1e-3), 'Heads produce identical queries'
""",
        },
    ],
}
