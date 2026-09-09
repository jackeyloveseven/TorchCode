"""2D Convolution task."""

TASK = {
    "title": "二维卷积",
    "difficulty": "Medium",
    "function_name": "my_conv2d",
    "hint": "使用 unfold 或嵌套循环提取图像块。对每个输出位置计算 sum(patch * kernel)。支持 stride 和 padding（使用 F.pad 进行零填充）。",
    "tests": [
        {
            "name": "输出形状",
            "code": "\nimport torch\nx = torch.randn(1, 3, 8, 8)\nw = torch.randn(16, 3, 3, 3)\nout = {fn}(x, w)\nassert out.shape == (1, 16, 6, 6), f'Shape: {out.shape}'\n"
        },
        {
            "name": "与 F.conv2d 结果一致",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.randn(2, 3, 8, 8)\nw = torch.randn(4, 3, 3, 3)\nb = torch.randn(4)\nout = {fn}(x, w, b)\nref = torch.nn.functional.conv2d(x, w, b)\nassert torch.allclose(out, ref, atol=1e-4), f'Max diff: {(out-ref).abs().max():.6f}'\n"
        },
        {
            "name": "带 padding",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.randn(1, 1, 5, 5)\nw = torch.randn(1, 1, 3, 3)\nout = {fn}(x, w, padding=1)\nref = torch.nn.functional.conv2d(x, w, padding=1)\nassert out.shape == ref.shape and torch.allclose(out, ref, atol=1e-4), 'Padding mismatch'\n"
        },
        {
            "name": "带 stride",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.randn(1, 1, 8, 8)\nw = torch.randn(1, 1, 3, 3)\nout = {fn}(x, w, stride=2)\nref = torch.nn.functional.conv2d(x, w, stride=2)\nassert out.shape == ref.shape and torch.allclose(out, ref, atol=1e-4), 'Stride mismatch'\n"
        },
        {
            "name": "梯度流",
            "code": "\nimport torch\nx = torch.randn(1, 1, 4, 4, requires_grad=True)\nw = torch.randn(2, 1, 3, 3, requires_grad=True)\n{fn}(x, w).sum().backward()\nassert x.grad is not None and w.grad is not None, 'Missing gradients'\n"
        }
    ]
}
