"""Implement Dropout task."""

TASK = {
    "title": "实现 Dropout",
    "difficulty": "Easy",
    "function_name": "MyDropout",
    "hint": "训练时：以概率 p 随机将元素置零，并将保留的元素缩放 1/(1-p)。推理时：恒等映射。使用 torch.rand_like 与 p 比较。",
    "tests": [
        {
            "name": "推理模式为恒等映射",
            "code": "\nimport torch, torch.nn as nn\nd = {fn}(p=0.5)\nassert isinstance(d, nn.Module), 'Must inherit from nn.Module'\nd.eval()\nx = torch.randn(4, 8)\nassert torch.equal(d(x), x), 'eval mode should return input unchanged'\n"
        },
        {
            "name": "训练模式：置零与缩放",
            "code": "\nimport torch\ntorch.manual_seed(42)\nd = {fn}(p=0.5)\nd.train()\nx = torch.ones(1000)\nout = d(x)\nassert (out == 0).any(), 'No zeros found during training'\nnon_zero = out[out != 0]\nassert torch.allclose(non_zero, torch.full_like(non_zero, 2.0), atol=1e-5), 'Non-zeros should be scaled by 1/(1-p)=2.0'\n"
        },
        {
            "name": "丢弃率约为 p",
            "code": "\nimport torch\ntorch.manual_seed(0)\nd = {fn}(p=0.3)\nd.train()\nout = d(torch.ones(10000))\nfrac = (out == 0).float().mean().item()\nassert 0.25 < frac < 0.35, f'Expected ~30%% zeros, got {frac*100:.1f}%%'\n"
        },
        {
            "name": "梯度流",
            "code": "\nimport torch\nd = {fn}(p=0.5)\nd.train()\nx = torch.randn(4, 8, requires_grad=True)\nd(x).sum().backward()\nassert x.grad is not None, 'x.grad is None'\n"
        }
    ]
}
