"""Gradient Accumulation task."""

TASK = {
    "title": "梯度累积",
    "difficulty": "Easy",
    "function_name": "accumulated_step",
    "hint": "只清零一次梯度。对每个 micro-batch：前向传播，loss/n_batches，反向传播。最后执行 optimizer.step()。损失缩放确保累积梯度与单个大批次等价。",
    "tests": [
        {
            "name": "与完整批次更新结果一致",
            "code": "\nimport torch, torch.nn as nn\ntorch.manual_seed(0)\nmodel = nn.Linear(4, 2, bias=False)\nmodel_ref = nn.Linear(4, 2, bias=False)\nmodel_ref.load_state_dict(model.state_dict())\nloss_fn = nn.MSELoss()\nopt = torch.optim.SGD(model.parameters(), lr=0.1)\nopt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.1)\nx1, y1 = torch.randn(2, 4), torch.randn(2, 2)\nx2, y2 = torch.randn(2, 4), torch.randn(2, 2)\n{fn}(model, opt, loss_fn, [(x1, y1), (x2, y2)])\nopt_ref.zero_grad()\nloss_ref = loss_fn(model_ref(torch.cat([x1, x2])), torch.cat([y1, y2]))\nloss_ref.backward()\nopt_ref.step()\nassert torch.allclose(model.weight.data, model_ref.weight.data, atol=1e-5), 'Must match full batch'\n"
        },
        {
            "name": "返回损失值",
            "code": "\nimport torch, torch.nn as nn\nmodel = nn.Linear(4, 2)\nopt = torch.optim.SGD(model.parameters(), lr=0.01)\nloss = {fn}(model, opt, nn.MSELoss(), [(torch.randn(2, 4), torch.randn(2, 2))])\nassert isinstance(loss, float), f'Should return float, got {type(loss)}'\nassert loss > 0, 'Loss should be positive'\n"
        },
        {
            "name": "参数确实更新",
            "code": "\nimport torch, torch.nn as nn\nmodel = nn.Linear(4, 2)\nopt = torch.optim.SGD(model.parameters(), lr=0.1)\nw_before = model.weight.data.clone()\n{fn}(model, opt, nn.MSELoss(), [(torch.randn(2, 4), torch.randn(2, 2))])\nassert not torch.equal(model.weight.data, w_before), 'Should change'\n"
        }
    ]
}
