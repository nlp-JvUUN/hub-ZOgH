"""
L0/L1 渐进式发现层：目录扫描 -> 元数据解析，且只解析「变化的部分」。

「渐进式加载」的第一层含义在这里落地：

  - L0：只列目录名（秒级），得到一个 skill 的"候选名单"；
  - L1：解析 SKILL.md 的 frontmatter，得到 SkillSpec 元数据；
  - 增量：manifest 缓存（state/manifest.json）记录每个 skill 的指纹
    （SKILL.md 的 md5 + mtime），再次扫描时未变化的 skill 直接命中缓存，
    只有新增/修改/删除的部分才真正重新解析 —— 扫描成本随变化量增长，
    而不是随 skill 总数增长（参考作业是每次全量重新解析所有 SKILL.md）。

Registry 只持有元数据（L1），实现代码（L2）与资源（L3）由 loader 懒加载，
discovery 永不 import 任何 skill 实现 —— 这是与参考作业的另一个差异：
他们加载阶段就完成了 frontmatter + 全量注册，我们把「能发现」和「能执行」
彻底分开，发现永远便宜。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .model import FrontmatterError, SkillSpec

# 约定：一个 skill 就是一个含 SKILL.md 的目录
SKILL_MD = "SKILL.md"


# ─────────────────────────────────────────────────────────────────────
# frontmatter 解析（零依赖的 YAML 子集）
# ─────────────────────────────────────────────────────────────────────

_SCALAR_RE = re.compile(
    r"^([A-Za-z0-9_.\-]+):\s*(.*)$"
)


def _parse_scalar(raw: str) -> Any:
    """解析一个标量/内联列表/内联字典。"""
    raw = raw.strip()
    if raw == "":
        return None
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    low = raw.lower()
    if low in ("null", "none", "~"):
        return None
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x.strip()) for x in _split_top_level(inner)]
    if raw.startswith("{") and raw.endswith("}"):
        inner = raw[1:-1].strip()
        out: Dict[str, Any] = {}
        if inner:
            for pair in _split_top_level(inner):
                k, _, v = pair.partition(":")
                out[k.strip().strip("\"'")] = _parse_scalar(v.strip())
        return out
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _split_top_level(s: str) -> List[str]:
    """按逗号切分，忽略引号内的逗号（用于内联列表/字典）。"""
    parts, buf, quote = [], [], None
    for ch in s:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def parse_frontmatter(text: str) -> Dict[str, Any]:
    """
    解析 SKILL.md 开头的 ```---\n...\n---``` 块。

    支持的语法（够用且自洽的 YAML 子集）：
      key: scalar                    字符串/数字/布尔/null/带引号字符串
      key: [a, b, c]                 内联列表
      key: {a: 1, b: two}            内联字典
      key:                           （下一级缩进更大的行构成嵌套块）
        child: value
      key: >-                        折叠多行字符串
        第一行
        第二行
    注释以 # 开头；纯空格缩进。
    """
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        raise FrontmatterError("SKILL.md 缺少 --- 包裹的 frontmatter 块")

    raw_lines: List[Tuple[int, str]] = []
    for raw in m.group(1).splitlines():
        if not raw.strip():
            continue
        if raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        raw_lines.append((indent, raw.strip()))

    result: Dict[str, Any] = {}
    pos = 0

    def parse_block(indent: int) -> Dict[str, Any]:
        nonlocal pos
        node: Dict[str, Any] = {}
        while pos < len(raw_lines):
            ind, line = raw_lines[pos]
            if ind < indent:
                break
            if ind > indent:
                raise FrontmatterError(f"意外的缩进: {line!r}")
            m2 = _SCALAR_RE.match(line)
            if not m2:
                raise FrontmatterError(f"无法解析的行: {line!r}")
            key, rest = m2.group(1), m2.group(2).strip()
            pos += 1
            if rest == ">-":  # 折叠多行字符串
                chunk: List[str] = []
                while pos < len(raw_lines) and raw_lines[pos][0] > indent:
                    chunk.append(raw_lines[pos][1])
                    pos += 1
                node[key] = " ".join(chunk)
            elif rest == "":
                if pos < len(raw_lines) and raw_lines[pos][0] > indent:
                    node[key] = parse_block(raw_lines[pos][0])
                else:
                    node[key] = None
            else:
                node[key] = _parse_scalar(rest)
        return node

    if raw_lines:
        result = parse_block(raw_lines[0][0])
    return result


# ─────────────────────────────────────────────────────────────────────
# 增量 manifest
# ─────────────────────────────────────────────────────────────────────


def _fingerprint(skill_md: Path) -> Tuple[str, int]:
    data = skill_md.read_bytes()
    return hashlib.md5(data).hexdigest(), skill_md.stat().st_mtime_ns


class Manifest:
    """
    SKILL.md 的增量扫描器（L1 层）。

    用法：manifest.scan() 返回 (specs, changed)；
    二次扫描时未变化的 skill 直接复用内存缓存，不重新读文件、不重新解析。
    """

    def __init__(self, skills_dir: Path, state_dir: Optional[Path] = None):
        self.skills_dir = Path(skills_dir)
        self.state_dir = Path(state_dir) if state_dir else self.skills_dir.parent / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, SkillSpec] = {}
        self._fingerprints: Dict[str, Tuple[str, int]] = {}
        self._load_cache_file()

    # ── 缓存文件 ──────────────────────────────────────────────

    def _cache_path(self) -> Path:
        return self.state_dir / "manifest.json"

    def _load_cache_file(self):
        """启动时从磁盘恢复上一次的指纹与规格（进程重启后依然增量）。"""
        p = self._cache_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for name, entry in data.items():
            try:
                spec = SkillSpec.from_dict(entry["spec"], skill_dir=entry.get("dir"))
            except Exception:
                continue
            self._cache[name] = spec
            self._fingerprints[name] = (entry["md5"], entry["mtime_ns"])

    def _save_cache_file(self):
        data = {}
        for name, spec in self._cache.items():
            fp = self._fingerprints.get(name)
            if fp is None:
                continue
            data[name] = {
                "md5": fp[0],
                "mtime_ns": fp[1],
                "spec": spec.to_dict(),
                "dir": str(spec.dir) if spec.dir else None,
            }
        self._cache_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    # ── 增量扫描 ──────────────────────────────────────────────

    def scan(self, force: bool = False) -> Tuple[List[SkillSpec], List[str]]:
        """
        扫描 skills 目录。

        Returns:
            (当前全部 specs, 本次发生变化的 skill 名列表)
        只重新解析 fingerprint 变化的目录；新增/删除也会被发现。
        """
        if not self.skills_dir.exists():
            return [], []
        dirs = sorted(
            d.name
            for d in self.skills_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        current = set(dirs)
        previous = set(self._cache.keys())

        changed: List[str] = []
        for name in sorted(current | previous):
            skill_dir = self.skills_dir / name
            md = skill_dir / SKILL_MD
            if name not in current:  # 被删除
                self._cache.pop(name, None)
                self._fingerprints.pop(name, None)
                changed.append(name)
                continue
            if not md.exists():
                self._cache.pop(name, None)
                self._fingerprints.pop(name, None)
                if name in previous:
                    changed.append(name)
                continue
            fp = _fingerprint(md)
            if not force and fp == self._fingerprints.get(name) and name in self._cache:
                continue  # 未变化：零解析成本
            try:
                spec = self._parse_skill(skill_dir)
            except FrontmatterError as e:
                # 解析失败：保留旧规格（如果存在），只记录变化
                if name in self._cache:
                    changed.append(name)
                continue
            self._cache[name] = spec
            self._fingerprints[name] = fp
            changed.append(name)

        self._save_cache_file()
        return self.list_all(), changed

    def _parse_skill(self, skill_dir: Path) -> SkillSpec:
        md = skill_dir / SKILL_MD
        text = md.read_text(encoding="utf-8")
        data = parse_frontmatter(text)
        spec = SkillSpec.from_dict(data, skill_dir=skill_dir)
        # 正文片段存入 notes，info 命令展示用
        body = text.split("---", 2)[-1].strip() if text.count("---") >= 2 else ""
        spec.notes = body[:300].strip()
        spec.md5, spec.mtime_ns = _fingerprint(md)
        return spec

    def list_all(self) -> List[SkillSpec]:
        return sorted(self._cache.values(), key=lambda s: s.name)

    def get(self, name: str) -> Optional[SkillSpec]:
        return self._cache.get(name)

    def reload_one(self, name: str) -> Optional[SkillSpec]:
        """强制重解析单个 skill（热更新入口）。"""
        spec = self._parse_skill(self.skills_dir / name)
        self._cache[name] = spec
        self._fingerprints[name] = (spec.md5, spec.mtime_ns)
        self._save_cache_file()
        return spec

    # ── 热更新轮询（watch 模式） ──────────────────────────────

    def watch(self, interval: float = 1.0, stop_event=None) -> Iterable[List[str]]:
        """
        轮询增量扫描，每当有变化就 yield 变化的 skill 名。
        这是「运行中渐进式加载」的入口：向 skills/ 丢一个新目录，
        harness 无需重启即可发现并执行它。
        """
        while True:
            if stop_event is not None and stop_event.is_set():
                return
            _, changed = self.scan()
            if changed:
                yield changed
            time.sleep(interval)


# ─────────────────────────────────────────────────────────────────────
# 注册表：只读元数据 + 依赖排序
# ─────────────────────────────────────────────────────────────────────


class Registry:
    """基于 Manifest 的只读注册表（L1 视图）。"""

    def __init__(self, manifest: Manifest):
        self.manifest = manifest

    def list_all(self) -> List[SkillSpec]:
        return self.manifest.list_all()

    def get(self, name: str) -> Optional[SkillSpec]:
        return self.manifest.get(name)

    def require(self, name: str) -> SkillSpec:
        spec = self.manifest.get(name)
        if spec is None:
            raise KeyError(f"skill 不存在: {name}（先运行 scan 或检查 skills/ 目录）")
        return spec

    def resolve_order(self, names: Iterable[str]) -> List[str]:
        """
        把请求的 skill 集合展开成「依赖先执行」的拓扑顺序。

        这里只对请求涉及的子图做 DFS 后序（附带环检测），
        不存在的依赖直接报错 —— 更符合「渐进」：只规划本次要跑的路径。
        """
        order: List[str] = []
        state: Dict[str, int] = {}  # 0=未访问 1=访问中 2=完成

        def visit(name: str):
            st = state.get(name, 0)
            if st == 2:
                return
            if st == 1:
                raise ValueError(f"循环依赖: {name}")
            state[name] = 1
            spec = self.get(name)
            if spec is None:
                raise ValueError(f"skill 不存在: {name}")
            for dep in spec.deps:
                visit(dep)
            state[name] = 2
            order.append(name)

        for name in names:
            visit(name)
        return order

    def check_deps(self) -> List[str]:
        """全量依赖体检，返回错误信息列表（scan/info 时展示）。"""
        errors = []
        for spec in self.list_all():
            for dep in spec.deps:
                if self.get(dep) is None:
                    errors.append(f"skill '{spec.name}' 依赖缺失: '{dep}'")
        return errors
