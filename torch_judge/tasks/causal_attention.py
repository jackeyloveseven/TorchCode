"""Causal Self-Attention task."""

TASK = {
    "title": "因果自注意力",
    "difficulty": "Hard",
    "function_name": "causal_attention",
    "hint": "与 softmax 注意力相同，但在 softmax 之前将未来位置掩码为 -inf。torch.triu(..., diagonal=1) 可获取上三角矩阵。",
    "tests": [
        {
            "name": "输出形状",
            "code": """
import torch
out = {fn}(torch.randn(2, 6, 16), torch.randn(2, 6, 16), torch.randn(2, 6, 16))
assert out.shape == (2, 6, 16), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "未来 token 不影响过去",
            "code": """
import torch
torch.manual_seed(0)
B, S, D = 1, 8, 16
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out1 = {fn}(Q, K, V)
K2, V2 = K.clone(), V.clone()
K2[:, 4:] = torch.randn(B, 4, D)
V2[:, 4:] = torch.randn(B, 4, D)
out2 = {fn}(Q, K2, V2)
assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5), 'Changing future K/V affected past outputs'
""",
        },
        {
            "name": "第一个位置只能看到自身",
            "code": """
import torch
torch.manual_seed(0)
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
out = {fn}(Q, K, V)
assert torch.allclose(out[:, 0], V[:, 0], atol=1e-5), 'Position 0 should output V[0]'
""",
        },
        {
            "name": "梯度流",
            "code": """
import torch
Q = torch.randn(2, 4, 8, requires_grad=True)
K = torch.randn(2, 4, 8, requires_grad=True)
V = torch.randn(2, 4, 8, requires_grad=True)
out = {fn}(Q, K, V)
out.sum().backward()
assert Q.grad is not None and K.grad is not None and V.grad is not None, 'Missing gradients'
""",
        },
    ],
}
