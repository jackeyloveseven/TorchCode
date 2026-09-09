"""Cosine LR Scheduler with Warmup task."""

TASK = {
    "title": "带 Warmup 的余弦学习率调度",
    "difficulty": "Medium",
    "function_name": "cosine_lr_schedule",
    "hint": "Warmup：在 warmup_steps 步内从 0 线性增长至 max_lr。之后余弦衰减：min_lr + 0.5*(max_lr-min_lr)*(1+cos(pi*progress))。",
    "tests": [
        {
            "name": "Warmup 起始",
            "code": "\nlr = {fn}(step=0, total_steps=100, warmup_steps=10, max_lr=0.001, min_lr=0.0)\nassert abs(lr) < 1e-8, f'lr at step 0: {lr}'\n"
        },
        {
            "name": "Warmup 结束",
            "code": "\nlr = {fn}(step=10, total_steps=100, warmup_steps=10, max_lr=0.001)\nassert abs(lr - 0.001) < 1e-8, f'lr at warmup end: {lr}'\n"
        },
        {
            "name": "调度结束",
            "code": "\nlr = {fn}(step=100, total_steps=100, warmup_steps=10, max_lr=0.001, min_lr=0.0001)\nassert abs(lr - 0.0001) < 1e-6, f'lr at end: {lr}'\n"
        },
        {
            "name": "Warmup 阶段单调递增",
            "code": "\nlrs = [{fn}(step=i, total_steps=100, warmup_steps=10, max_lr=0.001) for i in range(11)]\nfor i in range(len(lrs) - 1):\n    assert lrs[i] <= lrs[i+1] + 1e-10, f'Not increasing at step {i}'\n"
        },
        {
            "name": "余弦形状验证",
            "code": "\nimport math\nlr = {fn}(step=55, total_steps=100, warmup_steps=10, max_lr=0.001, min_lr=0.0)\nprogress = (55 - 10) / (100 - 10)\nexpected = 0.5 * 0.001 * (1 + math.cos(math.pi * progress))\nassert abs(lr - expected) < 1e-8, f'{lr} vs {expected}'\n"
        }
    ]
}
