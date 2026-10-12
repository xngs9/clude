# -*- coding: utf-8 -*-
"""
喵喵AI 的大脑 —— 从零手搓的一个小型神经网络。

不调用任何现成模型、不联网、零依赖（纯 Python 标准库）。
结构：字符/双字 特征 (哈希成固定维度) -> 隐藏层(tanh) -> 输出层(softmax)
训练：反向传播 + 随机梯度下降，梯度公式全部手写。

用法：
    from brain import Brain
    b = Brain()
    b.train()            # 在自带的小语料上训练
    b.save("brain.json")
    b.predict("把没用的清了")   # -> ("delete", 置信度)
"""
import json
import math
import random
import zlib

# 版本号：改动语料或结构就 +1，主程序据此判断要不要重新训练
DATASET_VER = 6

# ---------------------------------------------------------------- 训练语料
# 三个意图：delete（删）、create（建）、other（都不是）。

CORPUS = [
    # ---------- delete ----------
    ("删", "delete"), ("删除", "delete"), ("删掉", "delete"), ("删了", "delete"),
    ("把它删了", "delete"), ("删了它", "delete"), ("帮我删", "delete"), ("删除它", "delete"),
    ("把这个删了", "delete"), ("删掉这个", "delete"), ("给我删了", "delete"), ("删一下这个", "delete"),
    ("清", "delete"), ("清理", "delete"), ("清空", "delete"), ("清除", "delete"),
    ("清一下", "delete"), ("清掉", "delete"), ("清理一下", "delete"), ("清一清", "delete"),
    ("清扫", "delete"), ("打扫一下", "delete"), ("清理垃圾", "delete"), ("清空垃圾", "delete"),
    ("把垃圾清了", "delete"), ("把没用的删了", "delete"), ("清理临时文件", "delete"),
    ("删除没用的", "delete"), ("删掉没用的", "delete"), ("清理缓存", "delete"),
    ("删除日志", "delete"), ("清空回收站", "delete"), ("清理一下磁盘", "delete"),
    ("干掉", "delete"), ("干掉它", "delete"), ("干掉这个文件", "delete"),
    ("抹掉", "delete"), ("抹除", "delete"), ("消灭它", "delete"), ("删除文件", "delete"),
    ("把文件删了", "delete"), ("把那个删了", "delete"), ("不要了删了", "delete"),
    ("帮我清理一下没用的东西", "delete"), ("把不要的都删了", "delete"), ("把它清理掉", "delete"),
    ("remove", "delete"), ("delete", "delete"), ("del", "delete"), ("rm", "delete"),
    ("clean", "delete"), ("cleanup", "delete"), ("erase", "delete"),
    ("清理文件", "delete"), ("把文件清理掉", "delete"), ("删除这个文件", "delete"),
    ("清了这个文件", "delete"), ("文件删掉", "delete"), ("把这个破文件删了", "delete"),
    ("把没用的文件清了", "delete"), ("把那个文件清掉", "delete"), ("清理掉这个", "delete"),
    ("删了那个文件", "delete"), ("把桌面文件清了", "delete"), ("把那个破文件清了", "delete"),
    ("把我给你的文件清了", "delete"), ("顺手清掉", "delete"), ("清干净", "delete"),
    ("delete this file", "delete"), ("remove this file", "delete"), ("delete that file", "delete"),
    ("remove the file", "delete"), ("delete the file", "delete"), ("remove it", "delete"),
    ("delete it", "delete"), ("clear the files", "delete"), ("clean the files", "delete"),
    # ---------- create ----------
    ("创建", "create"), ("新建", "create"), ("建一个", "create"), ("建个", "create"),
    ("建一个文件", "create"), ("建个文件", "create"), ("新建文件", "create"),
    ("创建文件", "create"), ("新建文档", "create"), ("创建一个文档", "create"),
    ("创建文档", "create"), ("生成文件", "create"), ("生成一个文档", "create"),
    ("生成个文本", "create"), ("弄个记事本", "create"), ("弄一个记事本", "create"),
    ("加个文件", "create"), ("添加文件", "create"), ("创建一个txt", "create"),
    ("建个txt", "create"), ("新建文本文档", "create"), ("生成txt", "create"),
    ("造一个文件", "create"), ("造个文件", "create"), ("新建记事本", "create"),
    ("创建一个空文件", "create"), ("建一个空白文件", "create"), ("我要个新文件", "create"),
    ("给我建个文件", "create"), ("弄个文档", "create"), ("新建一个txt文件", "create"),
    ("创建一个文本", "create"), ("添加一个文档", "create"), ("写个空文本", "create"),
    ("帮我建一个文件", "create"), ("建个新的", "create"), ("新建一个", "create"),
    ("创建新的文件", "create"), ("弄一个文件出来", "create"), ("生成一个新的文本文件", "create"),
    ("建", "create"), ("create", "create"), ("create a file", "create"),
    ("new file", "create"), ("touch", "create"), ("make a file", "create"),
    ("搞个文件", "create"), ("来个文档", "create"), ("弄一份记事本", "create"),
    ("新建一个空白文档", "create"), ("给我弄个文本", "create"), ("创建个新文件", "create"),
    ("write a new file", "create"), ("make a new file", "create"),
    ("写", "create"), ("写一个", "create"), ("写个", "create"), ("写一个文件", "create"),
    ("写个文件", "create"), ("写文档", "create"), ("写个文档", "create"), ("写一篇文章", "create"),
    ("写周报", "create"), ("写记录", "create"), ("写个记录", "create"), ("写份记录", "create"),
    ("写一个会议记录", "create"), ("写个会议记录", "create"), ("写会议记录", "create"),
    ("会议记录", "create"), ("写一份报告", "create"), ("写报告", "create"), ("写总结", "create"),
    ("写个总结", "create"), ("帮我写点东西", "create"), ("帮我写个东西", "create"),
    ("帮我写一个", "create"), ("给我写一个", "create"), ("要一个会议记录", "create"),
    ("弄个会议记录", "create"), ("做个会议记录", "create"), ("待办清单", "create"),
    ("写日记", "create"), ("写个日记", "create"), ("写笔记", "create"), ("记一下", "create"),
    # ---------- other ----------
    ("你好", "other"), ("嗨", "other"), ("在吗", "other"), ("今天天气不错", "other"),
    ("你叫什么名字", "other"), ("你是谁", "other"), ("谢谢", "other"), ("哈哈", "other"),
    ("呵呵", "other"), ("滚", "other"), ("别烦我", "other"), ("几点了", "other"),
    ("1+1等于几", "other"), ("讲个笑话", "other"), ("喵", "other"), ("猫", "other"),
    ("天气真好", "other"), ("晚安", "other"), ("随便说说", "other"), ("我饿了", "other"),
    ("吃了吗", "other"), ("hello", "other"), ("hi", "other"), ("what time is it", "other"),
    ("无聊", "other"), ("睡觉了", "other"),
    ("good morning", "other"), ("good night", "other"), ("how are you", "other"),
    ("thanks", "other"), ("thank you", "other"), ("hello there", "other"),
    ("yes", "other"), ("no", "other"), ("ok", "other"), ("okay", "other"),
]

LABELS = ["delete", "create", "other"]

# 超参数
DIM = 256      # 哈希特征维度
HIDDEN = 32    # 隐藏层神经元
EPOCHS = 200
LR = 0.15


def featurize(text):
    """把一个短句变成稀疏特征下标集合：字符(1-gram) + 相邻双字(2-gram)，用 crc32 稳定哈希。"""
    t = text.strip().lower()
    idxs = set()
    for ch in t:
        if ch.strip():
            idxs.add(zlib.crc32(ch.encode("utf-8")) % DIM)
    for i in range(len(t) - 1):
        bg = t[i:i + 2]
        if bg.strip():
            idxs.add(zlib.crc32(bg.encode("utf-8")) % DIM)
    return idxs


class Brain:
    def __init__(self, seed=20261013):
        self.rng = random.Random(seed)
        r = lambda: self.rng.uniform(-0.1, 0.1)
        # W1: HIDDEN x DIM    W2: 3 x HIDDEN
        self.W1 = [[r() for _ in range(DIM)] for _ in range(HIDDEN)]
        self.b1 = [0.0] * HIDDEN
        self.W2 = [[r() for _ in range(HIDDEN)] for _ in range(len(LABELS))]
        self.b2 = [0.0] * len(LABELS)

    # -------- 前向 --------
    def _forward(self, idxs):
        h = [0.0] * HIDDEN
        for j in range(HIDDEN):
            s = self.b1[j]
            W1j = self.W1[j]
            for i in idxs:
                s += W1j[i]
            h[j] = math.tanh(s)
        o = [0.0] * len(LABELS)
        for k in range(len(LABELS)):
            s = self.b2[k]
            W2k = self.W2[k]
            for j in range(HIDDEN):
                s += W2k[j] * h[j]
            o[k] = s
        m = max(o)
        ex = [math.exp(v - m) for v in o]
        z = sum(ex)
        return h, [v / z for v in ex]

    # -------- 训练 --------
    def train(self, verbose=True):
        data = [(featurize(t), LABELS.index(y)) for t, y in CORPUS]
        for epoch in range(1, EPOCHS + 1):
            self.rng.shuffle(data)
            total = 0.0
            for idxs, y in data:
                h, p = self._forward(idxs)
                total += -math.log(max(p[y], 1e-9))
                # 输出层梯度: dL/do = p - onehot
                do = p[:]
                do[y] -= 1.0
                # 更新 W2 / b2
                for k in range(len(LABELS)):
                    W2k = self.W2[k]
                    g = do[k]
                    for j in range(HIDDEN):
                        W2k[j] -= LR * g * h[j]
                    self.b2[k] -= LR * g
                # 隐藏层梯度
                ds = [0.0] * HIDDEN
                for j in range(HIDDEN):
                    acc = 0.0
                    for k in range(len(LABELS)):
                        acc += self.W2[k][j] * do[k]
                    ds[j] = acc * (1.0 - h[j] * h[j])   # tanh 导数
                # 更新 W1 / b1（只动非零特征，稀疏）
                for j in range(HIDDEN):
                    W1j = self.W1[j]
                    g = LR * ds[j]
                    for i in idxs:
                        W1j[i] -= g
                    self.b1[j] -= g
            if verbose and (epoch % 25 == 0 or epoch == 1):
                print("  epoch %3d  loss=%.4f" % (epoch, total / len(data)))
            if total / len(data) < 0.01:
                if verbose:
                    print("  epoch %3d  loss=%.4f  (收敛，提前停)" % (epoch, total / len(data)))
                break

    # -------- 预测 --------
    def predict(self, text):
        _, p = self._forward(featurize(text))
        k = max(range(len(LABELS)), key=lambda i: p[i])
        return LABELS[k], p[k]

    # -------- 存取 --------
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ver": DATASET_VER, "dim": DIM, "hidden": HIDDEN,
                       "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("ver") != DATASET_VER or d.get("dim") != DIM or d.get("hidden") != HIDDEN:
            return None
        b = cls.__new__(cls)
        b.rng = random.Random(0)
        b.W1, b.b1, b.W2, b.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        return b


if __name__ == "__main__":
    import sys
    br = Brain()
    br.train()
    out = sys.argv[1] if len(sys.argv) > 1 else "brain.json"
    br.save(out)
    print("saved ->", out)
