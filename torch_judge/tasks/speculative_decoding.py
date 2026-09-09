"""Speculative Decoding task."""

TASK = {
    "title": "投机解码",
    "difficulty": "Hard",
    "function_name": "speculative_decode",
    "hint": "对每个草稿 token i：以概率 min(1, p_target[i,token]/p_draft[i,token]) 接受。若拒绝，则从归一化后的 max(0, p_target - p_draft) 中重新采样。返回已接受的 token 列表（可能包含一个重采样 token）。",
    "tests": [
        {
            "name": "完美草稿：全部接受",
            "code": "\nimport torch\ntorch.manual_seed(0)\nprobs = torch.softmax(torch.randn(4, 10), dim=-1)\ntokens = torch.tensor([2, 5, 1, 8])\naccepted = {fn}(probs, probs, tokens)\nassert len(accepted) == 4, f'Perfect draft should accept all, got {len(accepted)}'\nfor i in range(4):\n    assert accepted[i] == tokens[i].item(), f'Token {i} mismatch'\n"
        },
        {
            "name": "输出长度有界",
            "code": "\nimport torch\ntorch.manual_seed(0)\nK = 5\ntarget = torch.softmax(torch.randn(K, 8), dim=-1)\ndraft = torch.softmax(torch.randn(K, 8), dim=-1)\ntokens = torch.randint(0, 8, (K,))\naccepted = {fn}(target, draft, tokens)\nassert 1 <= len(accepted) <= K, f'Length {len(accepted)} not in [1, {K}]'\n"
        },
        {
            "name": "所有 token 合法",
            "code": "\nimport torch\nV = 8\nfor seed in range(20):\n    torch.manual_seed(seed)\n    target = torch.softmax(torch.randn(3, V), dim=-1)\n    draft = torch.softmax(torch.randn(3, V), dim=-1)\n    tokens = torch.randint(0, V, (3,))\n    for t in {fn}(target, draft, tokens):\n        assert 0 <= t < V, f'Token {t} out of range'\n"
        }
    ]
}
