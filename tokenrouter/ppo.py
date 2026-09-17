# -*- coding: utf-8 -*-
"""
PPO（近端策略优化）的纯 NumPy 实现

## 为什么不直接用 torch / stable-baselines3

1. **可复现优先于方便**：本项目全部立论建立在「同一颗种子跑出同一组数字」上。
   自实现的 PPO 只有一个随机源（numpy Generator），训练结果可精确复现；
   引入大型框架会带进 cuDNN 非确定性、多线程归约顺序等不可控因素。
2. **问题规模不需要它**：动作空间是 10 个候选模型，状态 9 维，一个隐藏层
   64 单元的网络在 CPU 上几十秒就能收敛，GPU 带来的只是麻烦。
3. **答辩时能被追问到底**：每一步（GAE、裁剪、熵正则）都在自己的代码里，
   不是一句「我们调了个库」。

## 实现要点

- 共享隐藏层的 actor-critic；动作屏蔽用掩码 softmax，保证不会采样到买不起的模型。
- 优势用 GAE(λ)，回报用 bootstrap，终止状态的价值按 0 处理。
- 更新用裁剪代理目标 + 价值损失 + 熵正则，Adam 分步。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """带掩码的 softmax（按行归一化），被屏蔽的动作概率为 0。"""
    neg = np.where(mask, logits, -np.inf)
    neg = np.where(np.isfinite(neg), neg, -1e30)
    z = neg - neg.max(axis=-1, keepdims=True)
    e = np.exp(z) * mask
    s = e.sum(axis=-1, keepdims=True)
    return np.where(s > 0, e / np.maximum(s, 1e-12), 1.0 / np.maximum(mask.sum(-1, keepdims=True), 1))


class ActorCritic:
    """共享隐藏层的小 MLP：策略头 + 价值头。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        s1 = 1.0 / np.sqrt(obs_dim)
        s2 = 1.0 / np.sqrt(hidden)
        self.W1 = rng.normal(0, s1, (obs_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, s2, (hidden, n_actions))
        self.b2 = np.zeros(n_actions)
        self.Wv = rng.normal(0, s2, (hidden, 1))
        self.bv = np.zeros(1)
        self.hidden = hidden

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H = np.tanh(X @ self.W1 + self.b1)
        logits = H @ self.W2 + self.b2
        v = (H @ self.Wv + self.bv)[:, 0]
        return H, logits, v

    def act(self, obs: np.ndarray, mask: np.ndarray,
            rng: np.random.default_rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """采样动作，返回 (动作, log π(a|s), V(s))。"""
        H, logits, v = self.forward(obs)
        p = masked_softmax(logits, mask)
        u = rng.random((p.shape[0], 1))
        c = np.cumsum(p, axis=-1)
        a = np.argmax(c > u, axis=-1)
        logp = np.log(np.take_along_axis(np.maximum(p, 1e-12), a[:, None], axis=1)[:, 0] + 1e-30)
        return a, logp, v

    def greedy(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """确定性动作（评估用）。"""
        _H, logits, _v = self.forward(obs)
        neg = np.where(mask, logits, -np.inf)
        neg = np.where(np.isfinite(neg), neg, -1e30)
        return np.argmax(neg, axis=-1)

    def logp_value(self, obs: np.ndarray, mask: np.ndarray,
                   a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H, logits, v = self.forward(obs)
        p = masked_softmax(logits, mask)
        logp = np.log(np.take_along_axis(np.maximum(p, 1e-30), a[:, None], axis=1)[:, 0])
        ent = -(p * np.log(np.maximum(p, 1e-30))).sum(-1)
        return logp, v, ent

    def params(self) -> List[str]:
        return ['W1', 'b1', 'W2', 'b2', 'Wv', 'bv']

    def grads(self, obs, mask, a, adv, ret, old_logp, clip_eps, vf_coef, ent_coef):
        """手工反向传播，返回**待最小化损失 L 的梯度**（Adam 做的是 param -= lr·∇L）。

        L = −min(r·A, clip(r)·A) − ent_coef·H + vf_coef·½(V−ret)²

        三处最容易写错、且写错后症状各不相同的地方，都在这里显式处理：

        - 策略项是 **负号**（最小化 −surrogate = 最大化 surrogate）。写成正号
          会让智能体反向学习：优势为正的动作被压低，训练曲线单调下行。
        - 链式法则要乘 **ratio**：surrogate 对 logp 求导是 r·A，不是 A。
        - softmax 的熵梯度是 −p·(log p + H)，**不是** −p·(log p + 1)。
        """
        B = obs.shape[0]
        H, logits, v = self.forward(obs)
        logp = np.log(np.take_along_axis(np.maximum(masked_softmax(logits, mask), 1e-30),
                                         a[:, None], axis=1)[:, 0])
        p = masked_softmax(logits, mask)

        ratio = np.exp(logp - old_logp)
        surr1 = ratio * adv
        surr2 = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        # 取 min 的那一支回传；落出裁剪区间时该支对 ratio 的导数为 0
        take1 = surr1 <= surr2
        g_ratio = np.where(take1, adv, np.where(
            (ratio > 1.0 - clip_eps) & (ratio < 1.0 + clip_eps), adv, 0.0))
        coef = ratio * g_ratio          # d/d logp = ratio · d/d ratio

        # d logp / d logits = 1{a} − p（softmax 的标准结论）
        onehot = np.zeros_like(p)
        onehot[np.arange(B), a] = 1.0
        dlogits_pi = -(onehot - p) * (coef / B)[:, None]

        ent = -(p * np.log(np.maximum(p, 1e-30))).sum(-1)
        dlogits_ent = ent_coef * (p * (np.log(np.maximum(p, 1e-30)) + ent[:, None])) / B

        dlogits = dlogits_pi + dlogits_ent
        dv = (vf_coef * (v - ret) / B)[:, None]

        dW2 = H.T @ dlogits
        db2 = dlogits.sum(0)
        dWv = H.T @ dv
        dbv = dv.sum(0)
        dH = dlogits @ self.W2.T + dv @ self.Wv.T
        dH = dH * (1.0 - H * H)                      # tanh'
        dW1 = obs.T @ dH
        db1 = dH.sum(0)
        g = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'Wv': dWv, 'bv': dbv}
        # 全局梯度裁剪：手工实现里最容易在训练后期被一个异常优势炸掉
        tot = sum(float((x * x).sum()) for x in g.values()) ** 0.5
        if tot > 5.0:
            s = 5.0 / tot
            g = {k: v * s for k, v in g.items()}
        return g


class Adam:
    def __init__(self, net: ActorCritic, lr: float = 3e-4):
        self.lr = lr
        self.t = 0
        self.m = {k: np.zeros_like(getattr(net, k)) for k in net.params()}
        self.v = {k: np.zeros_like(getattr(net, k)) for k in net.params()}

    def step(self, net: ActorCritic, grads: Dict[str, np.ndarray],
             b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.t += 1
        for k in net.params():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            setattr(net, k, getattr(net, k) - self.lr * mhat / (np.sqrt(vhat) + eps))


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                last_values: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """广义优势估计。rewards/values/dones 形状为 (T, n_envs)。"""
    T, N = rewards.shape
    adv = np.zeros((T, N))
    last = last_values.copy()
    gae = np.zeros(N)
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * last * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
        last = values[t]
    ret = adv + values
    return adv, ret


def collect_rollout(env, net: ActorCritic, rng: np.random.default_rng,
                    horizon: int) -> Dict[str, np.ndarray]:
    """跑一条完整轨迹（每条 env 一个 episode）。"""
    obs = env.reset()
    T = horizon
    N = env.n_envs
    A = env.m
    ob = np.zeros((T, N, obs.shape[1]))
    ac = np.zeros((T, N), dtype=np.int64)
    lp = np.zeros((T, N))
    vl = np.zeros((T, N))
    rw = np.zeros((T, N))
    dn = np.zeros((T, N))
    mk = np.zeros((T, N, A), dtype=bool)
    for t in range(T):
        mask = env.action_mask()
        a, logp, v = net.act(obs, mask, rng)
        ob[t] = obs
        mk[t] = mask
        ac[t] = a
        lp[t] = logp
        vl[t] = v
        obs, r, d, _ = env.step(a)
        rw[t] = r
        dn[t] = d
    _H, _l, last_v = net.forward(obs)
    last_values = last_v * (1.0 - dn[T - 1])
    return {'obs': ob, 'actions': ac, 'logp': lp, 'values': vl, 'rewards': rw,
            'dones': dn, 'masks': mk, 'last_values': last_values}


def train_ppo(env, net: ActorCritic, n_updates: int = 200, horizon: Optional[int] = None,
              gamma: float = 0.99, lam: float = 0.95, clip_eps: float = 0.2,
              vf_coef: float = 0.5, ent_coef: float = 0.01, lr: float = 3e-4,
              epochs: int = 4, minibatch: int = 256, seed: int = 0,
              log_every: int = 20) -> List[Dict]:
    """标准 PPO 训练循环。返回每轮日志（答对数 / 花费 / 损失）。"""
    rng = np.random.default_rng(seed)
    opt = Adam(net, lr=lr)
    horizon = horizon or (max(len(e) for e in env.episodes))
    logs: List[Dict] = []
    for u in range(n_updates):
        buf = collect_rollout(env, net, rng, horizon)
        adv, ret = compute_gae(buf['rewards'], buf['values'], buf['dones'],
                               buf['last_values'], gamma, lam)
        # 优势标准化：不标准化时 PPO 在这个奖励尺度上会很不稳定
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        T, N = buf['rewards'].shape
        flat = {k: buf[k].reshape(T * N, *buf[k].shape[2:])
                for k in ('obs', 'actions', 'logp', 'masks')}
        flat['adv'] = adv.reshape(T * N)
        flat['ret'] = ret.reshape(T * N)
        B = T * N
        idx = np.arange(B)
        losses = []
        for _ep in range(epochs):
            rng.shuffle(idx)
            for s in range(0, B, minibatch):
                sel = idx[s:s + minibatch]
                if len(sel) == 0:
                    continue
                g = net.grads(flat['obs'][sel], flat['masks'][sel], flat['actions'][sel],
                              flat['adv'][sel], flat['ret'][sel], flat['logp'][sel],
                              clip_eps, vf_coef, ent_coef)
                opt.step(net, g)
                _lp, v, ent = net.logp_value(flat['obs'][sel], flat['masks'][sel],
                                             flat['actions'][sel])
                losses.append(float(((v - flat['ret'][sel]) ** 2).mean()))
        if (u + 1) % log_every == 0 or u == 0:
            logs.append({
                'update': u + 1,
                'mean_correct': float(env.correct.mean()),
                'mean_spent': float(env.spent.mean()),
                'mean_served': float(env.served.mean()),
                'vf_loss': float(np.mean(losses)) if losses else 0.0,
            })
    return logs
