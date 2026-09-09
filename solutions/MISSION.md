# Mission: PyTorch 核心算子手撕能力

## Why
目标是通过逐题手撕 TorchCode 的 41 道题，系统掌握 PyTorch 中的核心算子、激活函数、注意力机制、归一化、优化器等关键模块的从零实现能力，从而通过 Meta、Google DeepMind、OpenAI 等顶级 ML 团队的白板编程面试。

## Success looks like
- 能在白板/面试中不借助文档，独立写出任意一道题的完整实现
- 理解每个算子的数学原理和 PyTorch autograd 机制
- 熟悉布尔掩码、einsum、矩阵乘法等 PyTorch 高效实现惯用法
- 41 道题全部通过 torch_judge 测试

## Constraints
- 按顺序从第 01 题逐题学习，不跳跃
- 每次一道题，彻底理解再进入下一道
- 中文教学

## Out of scope
- PyTorch 分布式训练（DDP、FSDP）
- 模型部署与推理优化（TorchScript、TensorRT）
- 数据加载与 DataLoader 优化
