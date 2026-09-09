"""Linear Self-Attention task."""

TASK = {
    "title": "线性自注意力",
    "difficulty": "Hard",
    "function_name": "linear_attention",
    "hint": "特征映射：phi(x) = elu(x) + 1。计算 phi(Q) @ (phi(K)^T @ V) 而非 softmax(Q @ K^T) @ V。用 phi(Q) @ sum(phi(K)) 归一化。",
    "tests": [
        {
            "name": "输出形状",
            "code": """
import torch
out = {fn}(torch.randn(2, 8, 16), torch.randn(2, 8, 16), torch.randn(2, 8, 32))
assert out.shape == (2, 8, 32), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "无 NaN 或 Inf",
            "code": """
import torch
torch.manual_seed(0)
out = {fn}(torch.randn(2, 16, 8), torch.randn(2, 16, 8), torch.randn(2, 16, 8))
assert not torch.isnan(out).any(), 'NaN in output'
assert not torch.isinf(out).any(), 'Inf in output'
""",
        },
        {
            "name": "梯度流",
            "code": """
import torch
Q = torch.randn(1, 4, 8, requires_grad=True)
K = torch.randn(1, 4, 8, requires_grad=True)
V = torch.randn(1, 4, 8, requires_grad=True)
{fn}(Q, K, V).sum().backward()
assert Q.grad is not None and K.grad is not None and V.grad is not None, 'Missing gradients'
""",
        },
        {
            "name": "长序列运行快速（线性复杂度）",
            "code": """
import torch, time
torch.manual_seed(0)
Q = torch.randn(1, 2048, 64)
K = torch.randn(1, 2048, 64)
V = torch.randn(1, 2048, 64)
t0 = time.perf_counter()
for _ in range(10):
    {fn}(Q, K, V)
elapsed = time.perf_counter() - t0
assert elapsed < 5.0, f'Too slow: {elapsed:.2f}s — should be O(S*D^2) not O(S^2*D)'
""",
        },
    ],
}
