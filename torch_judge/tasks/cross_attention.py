"""Multi-Head Cross-Attention task."""

TASK = {
    "title": "多头交叉注意力",
    "difficulty": "Medium",
    "function_name": "MultiHeadCrossAttention",
    "hint": "Q 来自解码器（x_q），K/V 来自编码器（x_kv）。投影后重塑为多头，计算缩放点积注意力（无因果掩码），拼接各头后做输出投影。",
    "tests": [
        {
            "name": "输出形状",
            "code": "\nimport torch, torch.nn as nn\nattn = {fn}(d_model=64, num_heads=4)\nassert isinstance(attn, nn.Module), 'Must inherit from nn.Module'\nout = attn(torch.randn(2, 6, 64), torch.randn(2, 10, 64))\nassert out.shape == (2, 6, 64), f'Output shape: {out.shape}'\n"
        },
        {
            "name": "Q 与 KV 长度不同",
            "code": "\nimport torch\nattn = {fn}(d_model=32, num_heads=2)\nout = attn(torch.randn(1, 3, 32), torch.randn(1, 20, 32))\nassert out.shape == (1, 3, 32), f'Shape: {out.shape}'\n"
        },
        {
            "name": "无因果掩码——所有 KV 影响所有 Q",
            "code": "\nimport torch\ntorch.manual_seed(0)\nattn = {fn}(d_model=32, num_heads=2)\nx_q = torch.randn(1, 4, 32)\nx_kv = torch.randn(1, 6, 32)\nout1 = attn(x_q, x_kv)\nx_kv2 = x_kv.clone()\nx_kv2[:, -1] = torch.randn(1, 32)\nout2 = attn(x_q, x_kv2)\nassert not torch.allclose(out1[:, 0], out2[:, 0], atol=1e-5), 'Changing last KV should affect all Q positions'\n"
        },
        {
            "name": "梯度流",
            "code": "\nimport torch\nattn = {fn}(d_model=32, num_heads=2)\nx_q = torch.randn(1, 4, 32, requires_grad=True)\nx_kv = torch.randn(1, 6, 32, requires_grad=True)\nattn(x_q, x_kv).sum().backward()\nassert x_q.grad is not None and x_kv.grad is not None, 'Missing gradients'\n"
        }
    ]
}
