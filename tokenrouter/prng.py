# -*- coding: utf-8 -*-
"""
与浏览器逐位一致的确定性随机源

为什么需要这个模块：
    控制台是零依赖的单文件 HTML，Token 侧的全部计算都要在浏览器里重算一遍。
    若 Python 与 JavaScript 生成的任务流对不上，控制台显示的数字就会与
    README / 实验结果不一致 —— 而「可复现」正是本项目唯一的立论基础。

    Python 的 random.Random 用 Mersenne Twister，且 randint / choices 内部
    依赖 getrandbits 与 bisect 的细节，几乎无法在 JS 里忠实复刻。
    因此两侧改用一个**可在 20 行内精确对拍的**算法：mulberry32。

移植要点（JS 语义的坑，改动前务必读）：
    1. 位运算前必须把操作数规范到 int32 / uint32：JS 会先 ToInt32/ToUint32，
       而 Python 对负数走无限二补码语义，仅在低 32 位上恰好一致。
    2. `t + x ^ t` 里的 `+` 是普通 number 加法，**不是** 32 位加法，
       必须手动取模后再做 XOR。
    3. `Math.imul` 是 32 位有符号乘法（取低 32 位）。
    4. `>>>` 是无符号右移，必须先转 uint32 再移位。
    5. `choices` 用「累积权重 + 一次 random() + 找第一个 cum > r」，
       与 CPython random.choices 的 bisect_right 语义等价；
       累积必须顺序相加，浮点顺序不能变。
"""

from __future__ import annotations

from typing import Sequence, TypeVar

_MASK32 = 0xFFFFFFFF

T = TypeVar('T')


def _u32(x: int) -> int:
    return x & _MASK32


def _s32(x: int) -> int:
    """转成 JS 的 int32 语义（有符号）。"""
    x &= _MASK32
    return x - (1 << 32) if x >= (1 << 31) else x


def _imul(x: int, y: int) -> int:
    """等价于 JS 的 Math.imul：32 位有符号乘法。"""
    return _s32(_s32(x) * _s32(y))


class Mulberry32:
    """mulberry32 伪随机数发生器，与 console/index.html 中的 JS 实现逐位一致。"""

    def __init__(self, seed: int):
        self.a = _s32(seed)

    def __call__(self) -> float:
        """返回 [0,1) 内的双精度浮点数。"""
        self.a = _s32(self.a + 0x6D2B79F5)
        t = _imul(self.a ^ (_u32(self.a) >> 15), 1 | self.a)
        inner = _imul(_u32(t) ^ (_u32(t) >> 7), 61 | t)
        t = _s32(((_u32(t) + _u32(inner)) & _MASK32) ^ _u32(t))
        return ((_u32(t) ^ (_u32(t) >> 14)) & _MASK32) / 4294967296.0

    # ---- 与 JS 端同名同语义的便捷方法 ----

    def random(self) -> float:
        return self()

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self()

    def randint(self, lo: int, hi: int) -> int:
        """闭区间 [lo, hi]。"""
        return lo + int(self() * (hi - lo + 1))

    def choices(self, population: Sequence[T], weights: Sequence[float]) -> T:
        """按权重抽一个元素，语义对齐 CPython 的 random.choices(k=1)。"""
        cum: list = []
        total = 0.0
        for w in weights:
            total += w
            cum.append(total)
        r = self() * total
        for item, c in zip(population, cum):
            if c > r:
                return item
        return population[-1]
