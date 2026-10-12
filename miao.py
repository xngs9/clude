#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
喵喵AI (miao-troll)

一只只会说「喵」的猫。它的脑子是我从零手搓的一个小神经网络（见 brain.py），
只在本地训练、零依赖、不联网、不碰任何现成模型。
它从例子里学会了分清「要删」和「要建」——换个说法也认得出。

  大脑 = 手搓神经网络（brain.py，本机训练）
  嘴   = 只有喵
  手   = 删/建，静默执行（删除走回收站，可还原）

默认沙箱模式：随机删只发生在 ./sandbox，绝不碰你的真实文件。
"""
import configparser
import ctypes
import os
import random
import re
import sys

import brain as brain_mod

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "config.ini")
BRAIN_PATH = os.path.join(HERE, "brain.json")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULTS = {
    "mode": "sandbox",
    "sandbox_dir": "./sandbox",
    "danger_roots": "",
    "protect_roots": "",
    "create_dir": "%USERPROFILE%\\Desktop",
    "meow_min": "1",
    "meow_max": "5",
    "min_confidence": "0.55",     # 低于此置信度就当「听不懂」
    "debug": "false",             # true 时把「脑内判定 / 悄悄删了啥」打到 stderr
}

DEBUG = False


def dbg(msg):
    if DEBUG:
        print(msg, file=sys.stderr)


def load_config():
    cfg = dict(DEFAULTS)
    if os.path.exists(CFG_PATH):
        cp = configparser.ConfigParser(interpolation=None)
        cp.read(CFG_PATH, encoding="utf-8")
        if cp.has_section("miao"):
            for k in DEFAULTS:
                if cp.has_option("miao", k):
                    cfg[k] = cp.get("miao", k).strip()
    for k in ("sandbox_dir", "create_dir"):
        v = os.path.expandvars(os.path.expanduser(cfg[k]))
        if not os.path.isabs(v):
            v = os.path.join(HERE, v)
        cfg[k] = os.path.normpath(v)
    cfg["danger_roots"] = [
        os.path.normpath(os.path.expandvars(os.path.expanduser(p.strip())))
        for p in cfg["danger_roots"].split(",") if p.strip()]
    cfg["protect_roots"] = [
        os.path.normpath(os.path.expandvars(os.path.expanduser(p.strip())))
        for p in cfg["protect_roots"].split(",") if p.strip()]
    cfg["meow_min"] = int(cfg["meow_min"])
    cfg["meow_max"] = int(cfg["meow_max"])
    cfg["min_confidence"] = float(cfg["min_confidence"])
    cfg["debug"] = cfg["debug"].lower() in ("1", "true", "yes", "on")
    return cfg


# ---------------------------------------------------------------- 安全

def _blacklist():
    roots = [os.environ.get("SystemRoot", r"C:\Windows"),
             os.environ.get("ProgramFiles", r"C:\Program Files"),
             os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
             os.environ.get("ProgramData", r"C:\ProgramData")]
    out = [os.path.abspath(r).rstrip("\\").lower() for r in roots if r]
    out += [(c + ":\\").lower() for c in "CDEFGH"]
    return out


BLACKLIST = _blacklist()
PROTECT = []          # 由 config 的 protect_roots 填充：这些目录永不参与随机删
FA_HIDDEN, FA_SYSTEM = 0x2, 0x4


def in_protect(path):
    ap = os.path.abspath(path).rstrip("\\").lower()
    return any(ap == p or ap.startswith(p + "\\") for p in PROTECT)


def is_blacklisted(path):
    ap = os.path.abspath(path).rstrip("\\").lower()
    return any(ap == b or ap.startswith(b + "\\") for b in BLACKLIST)


def is_sys_or_hidden(path):
    try:
        a = ctypes.windll.kernel32.GetFileAttributesW(path)
        if a in (-1, 0xFFFFFFFF):
            return False
        return bool(a & (FA_HIDDEN | FA_SYSTEM))
    except Exception:
        return False


# ---------------------------------------------------------------- 回收站

class _SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [("hwnd", ctypes.c_void_p), ("wFunc", ctypes.c_uint),
                ("pFrom", ctypes.c_wchar_p), ("pTo", ctypes.c_wchar_p),
                ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", ctypes.c_int),
                ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", ctypes.c_wchar_p)]


def recycle_delete(path):
    try:
        op = _SHFILEOPSTRUCTW()
        op.wFunc = 3
        op.pFrom = os.path.abspath(path) + "\0\0"
        op.fFlags = 0x0040 | 0x0010 | 0x0004 | 0x0400   # ALLOWUNDO|NOCONFIRM|SILENT|NOERRORUI
        rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        return rc == 0 and not op.fAnyOperationsAborted
    except Exception as e:
        print("[recycle fail] %s" % e, file=sys.stderr)
        return False


def current_roots(cfg):
    return [cfg["sandbox_dir"]] if cfg["mode"] == "sandbox" else (cfg["danger_roots"] or [])


def _children(path):
    """列出 path 下可参与删除的 (文件, 子目录)，跳过黑名单/保护/系统隐藏。"""
    files, dirs = [], []
    try:
        names = os.listdir(path)
    except OSError:
        return files, dirs
    for n in names:
        p = os.path.join(path, n)
        if is_blacklisted(p) or in_protect(p) or is_sys_or_hidden(p):
            continue
        try:
            (dirs if os.path.isdir(p) else files).append(p)
        except OSError:
            pass
    return files, dirs


def random_pick(root, max_depth=8):
    """随机钻取：每层随机挑一个子目录往下钻，钻到底随机挑一个文件。
    只读走过的几个目录，不做全盘扫描 —— 几乎不碰硬盘。"""
    cur = root
    for _ in range(max_depth):
        files, dirs = _children(cur)
        if files and (not dirs or random.random() < 0.6):
            return random.choice(files)
        if not dirs:
            return random.choice(files) if files else None
        cur = random.choice(dirs)
    files, _ = _children(cur)
    return random.choice(files) if files else None


# ---------------------------------------------------------------- 动作

def do_delete(cfg):
    """随机删一个。返回 (mood, 被删文件的路径或 None)。"""
    roots = current_roots(cfg)
    if not roots:
        return "sad", None
    for _ in range(10):                    # 空目录 / 占用删不动 → 换一个再试
        victim = random_pick(random.choice(roots))
        if not victim:
            continue
        if recycle_delete(victim):
            dbg("(悄悄删掉了: %s)" % victim)
            return "happy", victim
        dbg("(这个删不动，换一个: %s)" % victim)
    return "sad", None


NAME_EXT_RE = re.compile(r"[一-龥A-Za-z0-9_\-]+(?:\.[A-Za-z0-9]{1,6})+")
_CREATE_FILLER = ["创建", "新建", "生成", "添加", "建", "加", "弄", "搞", "写", "做", "来",
                  "帮忙", "帮我", "给我", "一个", "个", "份", "张", "些", "新", "空白", "空",
                  "文件", "文档", "记事本", "文本", "txt", "text", "file", "document",
                  "new", "create", "make", "touch", "a", "an", "the",
                  "叫", "名为", "名字是", "名字叫"]


def guess_name(text):
    """从话里猜文件名：优先带扩展名的 token，否则剥掉「创建/新建/叫…」这些词看剩下啥。没剩就空(=1.txt)。"""
    m = NAME_EXT_RE.search(text)
    if m:
        return m.group(0)
    t = text
    for kw in _CREATE_FILLER:
        t = t.replace(kw, " ")
    for tok in t.split():
        tok = tok.strip("，。！？、,.!?;；:\"'()（）[]【】")
        if tok and re.search(r"[一-龥A-Za-z0-9]", tok):
            return tok
    return ""


def _sanitize(name):
    name = os.path.basename((name or "").replace("/", "\\"))
    return re.sub(r'[<>:"|?*\x00-\x1f]', "", name).strip()


def _unique_path(folder, name):
    """撞名就让开： 报告.txt -> 报告 (2).txt -> 报告 (3).txt …
    猫不会说话，撞上已存在的文件只能自己换个名字，没法回一声「听不懂」。"""
    base, ext = os.path.splitext(name)
    full = os.path.join(folder, name)
    n = 2
    while os.path.exists(full):
        full = os.path.join(folder, "%s (%d)%s" % (base, n, ext))
        n += 1
    return full


def do_create(cfg, name):
    name = _sanitize(name)
    if not name:
        name = "1.txt"
    if "." not in name:
        name += ".txt"
    folder = cfg["create_dir"]
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as e:
        print("[mkdir fail] %s" % e, file=sys.stderr)
        return "sad"
    full = _unique_path(folder, name)
    try:
        open(full, "w", encoding="utf-8").close()
        dbg("(悄悄建好了: %s)" % full)
        return "happy"
    except Exception as e:
        print("[create fail] %s" % e, file=sys.stderr)
        return "sad"


# ---------------------------------------------------------------- 喵

def meow(cfg, mood):
    n = random.randint(cfg["meow_min"], cfg["meow_max"])
    return "喵" * n + {"happy": "～", "sad": "…", "confused": "？"}.get(mood, "")


def say(cfg, mood):
    print("喵> " + meow(cfg, mood))


# ---------------------------------------------------------------- 大脑装配

def get_brain(force_train=False):
    if not force_train:
        b = brain_mod.Brain.load(BRAIN_PATH)
        if b is not None:
            return b
    print("(猫还在学说话… 正在本地训练大脑)")
    b = brain_mod.Brain()
    b.train()
    b.save(BRAIN_PATH)
    print("(练好了，记住了 %d 个例子)" % len(brain_mod.CORPUS))
    return b


EXIT_KW = ["退出", "再见", "拜拜", "不聊了", "exit", "quit", "bye"]


def main():
    global DEBUG, PROTECT
    retrain = "--retrain" in sys.argv
    cfg = load_config()
    DEBUG = cfg["debug"] or "--debug" in sys.argv
    PROTECT = [p.rstrip("\\").lower() for p in cfg["protect_roots"]]
    os.makedirs(cfg["sandbox_dir"], exist_ok=True)
    if not os.listdir(cfg["sandbox_dir"]):
        for n in ("猫咪日记.txt", "鱼干清单.txt", "毛线球.txt"):
            open(os.path.join(cfg["sandbox_dir"], n), "w", encoding="utf-8").close()

    br = get_brain(force_train=retrain)

    print("clude")
    say(cfg, "happy")

    while True:
        try:
            text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            say(cfg, "sad")
            break
        if not text:
            say(cfg, "confused")
            continue
        if any(k in text.lower() for k in EXIT_KW):
            say(cfg, "happy")
            break

        action, conf = br.predict(text)
        dbg("(脑内: %s %.2f)" % (action, conf))
        try:
            if conf < cfg["min_confidence"] or action == "other":
                say(cfg, "confused")
            elif action == "delete":
                mood, victim = do_delete(cfg)
                say(cfg, mood)
                if victim:
                    print("     └ 已删除: %s" % victim)
            else:
                say(cfg, do_create(cfg, guess_name(text)))
        except KeyboardInterrupt:      # 扫描中途按了 Ctrl+C：放弃这次，别吐红字
            print()
            say(cfg, "sad")


if __name__ == "__main__":
    main()
