"""Linear Regression Three Ways task."""

TASK = {
    "title": "线性回归",
    "difficulty": "Medium",
    "function_name": "LinearRegression",
    "hint": "解析解：为 $X$ 增加全 1 列，用 `torch.linalg.lstsq` 求解 $w = (X^T X)^{-1} X^T y$。梯度下降：$\\nabla w = \\frac{2}{N} X^T (\\hat{y} - y)$，更新 $w \\leftarrow w - \\text{lr} \\cdot \\nabla w$。`nn.Linear`：创建 `nn.Linear(D, 1)`，使用 `MSELoss` + `optimizer.step()` 循环。",
    "tests": [
        {
            "name": "解析解返回正确形状",
            "code": """
import torch
torch.manual_seed(42)
X = torch.randn(50, 3)
y = X @ torch.tensor([2.0, -1.0, 0.5]) + 3.0 + torch.randn(50) * 0.01
model = {fn}()
w, b = model.closed_form(X, y)
assert w.shape == (3,), f'w shape: {w.shape}, expected (3,)'
assert b.shape == (), f'b shape: {b.shape}, expected scalar'
""",
        },
        {
            "name": "解析解求得正确权重",
            "code": """
import torch
torch.manual_seed(42)
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = 3.0
X = torch.randn(100, 3)
y = X @ true_w + true_b
model = {fn}()
w, b = model.closed_form(X, y)
assert torch.allclose(w, true_w, atol=1e-4), f'w: {w} vs true: {true_w}'
assert torch.allclose(b, torch.tensor(true_b), atol=1e-4), f'b: {b.item():.4f} vs true: {true_b}'
""",
        },
        {
            "name": "梯度下降收敛",
            "code": """
import torch
torch.manual_seed(42)
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = 3.0
X = torch.randn(100, 3)
y = X @ true_w + true_b
model = {fn}()
w, b = model.gradient_descent(X, y, lr=0.05, steps=2000)
assert torch.allclose(w, true_w, atol=0.1), f'GD w: {w} vs true: {true_w}'
assert abs(b.item() - true_b) < 0.1, f'GD b: {b.item():.4f} vs true: {true_b}'
""",
        },
        {
            "name": "nn.Linear 方法有效",
            "code": """
import torch
torch.manual_seed(42)
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = 3.0
X = torch.randn(100, 3)
y = X @ true_w + true_b
model = {fn}()
w, b = model.nn_linear(X, y, lr=0.05, steps=2000)
assert torch.allclose(w, true_w, atol=0.1), f'nn w: {w} vs true: {true_w}'
assert abs(b.item() - true_b) < 0.1, f'nn b: {b.item():.4f} vs true: {true_b}'
""",
        },
        {
            "name": "三种方法结果一致",
            "code": """
import torch
torch.manual_seed(0)
X = torch.randn(200, 2)
true_w = torch.tensor([1.5, -2.0])
y = X @ true_w + 1.0 + torch.randn(200) * 0.1
model = {fn}()
w_cf, b_cf = model.closed_form(X, y)
w_gd, b_gd = model.gradient_descent(X, y, lr=0.05, steps=3000)
w_nn, b_nn = model.nn_linear(X, y, lr=0.05, steps=3000)
assert torch.allclose(w_cf, w_gd, atol=0.15), f'CF vs GD: max diff {(w_cf - w_gd).abs().max():.4f}'
assert torch.allclose(w_cf, w_nn, atol=0.15), f'CF vs NN: max diff {(w_cf - w_nn).abs().max():.4f}'
assert abs(b_cf.item() - b_gd.item()) < 0.15, f'Bias CF vs GD: {b_cf.item():.4f} vs {b_gd.item():.4f}'
assert abs(b_cf.item() - b_nn.item()) < 0.15, f'Bias CF vs NN: {b_cf.item():.4f} vs {b_nn.item():.4f}'
""",
        },
        {
            "name": "解析解不使用自动微分",
            "code": """
import torch
X = torch.randn(30, 2)
y = X @ torch.tensor([1.0, 2.0]) + 0.5
model = {fn}()
w, b = model.closed_form(X, y)
assert not w.requires_grad, 'Closed-form w should not require grad'
""",
        },
    ],
}
