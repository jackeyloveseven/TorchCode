"""Gradient Norm Clipping task."""

TASK = {
    "title": "梯度范数裁剪",
    "difficulty": "Easy",
    "function_name": "clip_grad_norm",
    "hint": "总范数 = sqrt(sum(p.grad.norm()^2))。若总范数 > max_norm，将所有梯度缩放 max_norm/total。返回原始总范数。",
    "tests": [
        {
            "name": "裁剪至 max_norm",
            "code": "\nimport torch\np1 = torch.randn(10, requires_grad=True)\np2 = torch.randn(10, requires_grad=True)\n(p1 * 10).sum().backward()\n(p2 * 10).sum().backward()\n{fn}([p1, p2], max_norm=1.0)\nnew_norm = torch.sqrt(p1.grad.norm()**2 + p2.grad.norm()**2).item()\nassert new_norm <= 1.0 + 1e-5, f'Clipped norm {new_norm:.4f} > 1.0'\n"
        },
        {
            "name": "返回原始范数",
            "code": "\nimport torch\np = torch.randn(10, requires_grad=True)\n(p * 3).sum().backward()\nexpected = p.grad.norm().item()\nreturned = {fn}([p], max_norm=100.0)\nassert abs(returned - expected) < 1e-4, f'Returned {returned:.4f}, expected {expected:.4f}'\n"
        },
        {
            "name": "范数小于 max_norm 时不裁剪",
            "code": "\nimport torch\np = torch.randn(4, requires_grad=True)\n(p * 0.001).sum().backward()\ngrad_before = p.grad.clone()\n{fn}([p], max_norm=100.0)\nassert torch.equal(p.grad, grad_before), 'Should not change when norm < max_norm'\n"
        },
        {
            "name": "保持方向",
            "code": "\nimport torch\ntorch.manual_seed(0)\np = torch.randn(100, requires_grad=True)\n(p * 10).sum().backward()\ndir_before = p.grad / p.grad.norm()\n{fn}([p], max_norm=1.0)\ndir_after = p.grad / p.grad.norm()\nassert torch.allclose(dir_before, dir_after, atol=1e-5), 'Should preserve direction'\n"
        }
    ]
}
