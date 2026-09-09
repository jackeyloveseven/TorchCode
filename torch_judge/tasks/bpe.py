"""Byte-Pair Encoding (BPE) task."""

TASK = {
    "title": "字节对编码（BPE）",
    "difficulty": "Hard",
    "function_name": "SimpleBPE",
    "hint": "train：将单词拆分为字符 + </w>，迭代寻找最高频的相邻对并合并。encode：按学到的合并顺序将文本拆分为子词。",
    "tests": [
        {
            "name": "合并次数正确",
            "code": "\nbpe = {fn}()\nbpe.train(['low', 'low', 'low', 'lower', 'newest', 'widest'], num_merges=5)\nassert len(bpe.merges) == 5, f'Expected 5 merges, got {len(bpe.merges)}'\n"
        },
        {
            "name": "最高频对最先合并",
            "code": "\nbpe = {fn}()\nbpe.train(['aaa', 'aaa', 'aaa', 'bbb'], num_merges=1)\nassert bpe.merges[0] == ('a', 'a'), f'First merge: {bpe.merges[0]}'\n"
        },
        {
            "name": "encode 返回字符串列表",
            "code": "\nbpe = {fn}()\nbpe.train(['low', 'lower', 'lowest'] * 3, num_merges=10)\ntokens = bpe.encode('low')\nassert isinstance(tokens, list), 'encode must return a list'\nassert all(isinstance(t, str) for t in tokens), 'tokens must be strings'\nreconstructed = ''.join(t.replace('</w>', '') for t in tokens)\nassert reconstructed == 'low', f'Reconstruction: {reconstructed}'\n"
        },
        {
            "name": "更多合并 -> 更少 token",
            "code": "\nbpe1 = {fn}()\nbpe1.train(['hello'] * 10, num_merges=2)\nbpe2 = {fn}()\nbpe2.train(['hello'] * 10, num_merges=10)\nassert len(bpe2.encode('hello')) <= len(bpe1.encode('hello')), 'More merges should reduce tokens'\n"
        }
    ]
}
