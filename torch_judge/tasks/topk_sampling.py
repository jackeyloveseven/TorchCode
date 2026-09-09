"""Top-k / Top-p Sampling task."""

TASK = {
    "title": "Top-k / Top-p 采样",
    "difficulty": "Medium",
    "function_name": "sample_top_k_top_p",
    "hint": "先应用温度缩放。Top-k：将排名在第 k 之后的 logits 设为 -inf。Top-p：排序后计算概率累积和，将累积和超过 p 的位置掩码。最后从 softmax 中采样。",
    "tests": [
        {
            "name": "top_k=1 始终返回 argmax",
            "code": "\nimport torch\ntorch.manual_seed(0)\nlogits = torch.tensor([1.0, 5.0, 2.0, 0.5])\nfor _ in range(10):\n    assert {fn}(logits.clone(), top_k=1) == 1, 'top_k=1 should return argmax'\n"
        },
        {
            "name": "低温度集中分布",
            "code": "\nimport torch\ntorch.manual_seed(42)\nlogits = torch.tensor([1.0, 3.0, 2.0])\ncounts = [0, 0, 0]\nfor _ in range(100):\n    counts[{fn}(logits.clone(), temperature=0.01)] += 1\nassert counts[1] > 90, f'Low temp should pick argmax, got {counts}'\n"
        },
        {
            "name": "所有 token 可达（无过滤）",
            "code": "\nimport torch\nlogits = torch.zeros(5)\nseen = set()\nfor i in range(200):\n    torch.manual_seed(i)\n    seen.add({fn}(logits.clone()))\nassert len(seen) == 5, f'Only saw {seen}'\n"
        },
        {
            "name": "返回合法索引",
            "code": "\nimport torch\ntorch.manual_seed(0)\nV = 100\nlogits = torch.randn(V)\nfor _ in range(20):\n    t = {fn}(logits.clone(), top_k=10, top_p=0.9)\n    assert 0 <= t < V, f'Token {t} out of range'\n"
        }
    ]
}
