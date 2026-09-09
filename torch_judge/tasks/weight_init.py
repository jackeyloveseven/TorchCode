"""Kaiming Initialization task."""

TASK = {
    "title": "Kaiming 初始化",
    "difficulty": "Easy",
    "function_name": "kaiming_init",
    "hint": "fan_in 模式：std = sqrt(2 / fan_in)，其中 fan_in = weight.shape[1]。用 normal(0, std) 填充张量并返回。",
    "tests": [
        {
            "name": "均值近似为 0",
            "code": "\nimport torch\ntorch.manual_seed(0)\nw = torch.empty(256, 512)\n{fn}(w)\nassert abs(w.mean().item()) < 0.02, f'Mean too far from 0: {w.mean().item():.4f}'\n"
        },
        {
            "name": "标准差匹配 sqrt(2/fan_in)",
            "code": "\nimport torch, math\ntorch.manual_seed(0)\nfan_in = 1024\nw = torch.empty(256, fan_in)\n{fn}(w)\nexpected = math.sqrt(2.0 / fan_in)\nassert abs(w.std().item() - expected) < 0.005, f'Std {w.std().item():.4f} vs expected {expected:.4f}'\n"
        },
        {
            "name": "返回同一张量（原地操作）",
            "code": "\nimport torch\nw = torch.empty(64, 32)\nout = {fn}(w)\nassert out is w, 'Should return the same tensor'\nassert out.shape == (64, 32), 'Shape should be unchanged'\n"
        },
        {
            "name": "更小的 fan_in 产生更大标准差",
            "code": "\nimport torch\nw1 = torch.empty(64, 16)\nw2 = torch.empty(64, 256)\n{fn}(w1)\n{fn}(w2)\nassert w1.std().item() > w2.std().item(), 'Smaller fan_in should give larger std'\n"
        }
    ]
}
