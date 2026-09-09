# TorchCode PyTorch 学习资源

## Knowledge

- [PyTorch 官方文档 · torch.Tensor](https://pytorch.org/docs/stable/tensors.html)
  张量 API 权威参考。用于：查询各算子的签名、参数、支持的数据类型。

- [PyTorch 官方文档 · Autograd 机制](https://pytorch.org/docs/stable/notes/autograd.html)
  解释计算图、反向传播、requires_grad。用于：理解为什么某些实现方式支持/不支持梯度。

- [PyTorch 官方文档 · torch.nn.functional](https://pytorch.org/docs/stable/nn.functional.html)
  内置激活函数、归一化、损失函数的参考实现源码。用于：对比自己实现与官方实现的差异。

- [Stanford CS231n Lecture Notes](http://cs231n.github.io/neural-networks-1/)
  激活函数、归一化的直觉解释。用于：理解 ReLU、BatchNorm 等为什么在实践中有效。

- [Attention Is All You Need (原始论文)](https://arxiv.org/abs/1706.03762)
  Transformer 架构的一手来源。用于：05~14 题（注意力机制系列）的数学推导参考。

- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)
  IO-aware exact attention 算法。用于：第 25 题 flash_attention。

## Wisdom (Communities)

- [PyTorch 论坛 (discuss.pytorch.org)](https://discuss.pytorch.org)
  官方社区，高质量技术问答。用于：遇到奇怪 autograd 行为或性能问题时搜索。

- [r/MachineLearning](https://www.reddit.com/r/MachineLearning)
  ML 研究社区，学术讨论质量高。用于：了解某个算子的设计动机和最新进展。
