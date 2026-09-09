"""Embedding Layer task."""

TASK = {
    "title": "Embedding 层",
    "difficulty": "Easy",
    "function_name": "MyEmbedding",
    "hint": "将形状为 (num_embeddings, embedding_dim) 的权重矩阵存储为 nn.Parameter。前向传播 = weight[indices]。",
    "tests": [
        {
            "name": "权重形状",
            "code": "\nimport torch, torch.nn as nn\nemb = {fn}(100, 32)\nassert isinstance(emb, nn.Module), 'Must inherit from nn.Module'\nassert hasattr(emb, 'weight'), 'Need self.weight'\nassert emb.weight.shape == (100, 32), f'Weight shape: {emb.weight.shape}'\nassert emb.weight.requires_grad, 'weight must require grad'\n"
        },
        {
            "name": "查表正确性",
            "code": "\nimport torch\nemb = {fn}(10, 4)\nidx = torch.tensor([0, 3, 7])\nout = emb(idx)\nassert out.shape == (3, 4), f'Output shape: {out.shape}'\nassert torch.equal(out[0], emb.weight[0]), 'Mismatch at index 0'\nassert torch.equal(out[1], emb.weight[3]), 'Mismatch at index 3'\n"
        },
        {
            "name": "批量索引",
            "code": "\nimport torch\nemb = {fn}(20, 8)\nidx = torch.tensor([[1, 2], [3, 4]])\nout = emb(idx)\nassert out.shape == (2, 2, 8), f'Batch output shape: {out.shape}'\n"
        },
        {
            "name": "梯度流",
            "code": "\nimport torch\nemb = {fn}(10, 4)\nout = emb(torch.tensor([2, 5]))\nout.sum().backward()\nassert emb.weight.grad is not None, 'weight.grad is None'\nassert emb.weight.grad[2].abs().sum() > 0, 'Grad at used index should be non-zero'\nassert emb.weight.grad[0].abs().sum() == 0, 'Grad at unused index should be zero'\n"
        }
    ]
}
