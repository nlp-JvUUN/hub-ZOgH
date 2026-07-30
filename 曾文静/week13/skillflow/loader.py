"""
L2/L3 渐进式加载层：实现代码与资源都「用到才加载」，并受加载预算约束。

「渐进式加载」的第二层含义在这里落地：

  - L2：skill.py 只在第一次真正执行时 import（懒加载），加载后缓存；
        实现文件被修改后再次执行时自动重新加载（热更新），无需重启进程。
  - L3：resources/ 里的数据文件只在 skill 运行中明确请求时才读入内存，
        且提供「先看清单、再按需取用」的接口（list_resources 不读内容）。
  - 预算：每个 skill 在 SKILL.md 里声明 weight（加载代价）。LoadBudget 限制
    本次进程最多加载多少 weight 的实现；超预算的 skill 不加载、不执行，
    由引擎发出 stage_defer 事件 —— 「渐进」意味着成本是显式可控的。

这里 L2/L3 与 L1 彻底分离，一个 skill 被发现（discovery）后，
可能在整个进程生命周期里从未被加载过 —— 加载量 = 实际使用量。
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .model import BudgetExceeded, SkillSpec


class LoadBudget:
    """实现加载的预算账本。"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.spent = 0

    @property
    def remaining(self) -> int:
        return max(0, self.capacity - self.spent)

    def check(self, spec: SkillSpec) -> bool:
        return self.spent + spec.weight <= self.capacity

    def spend(self, spec: SkillSpec):
        if not self.check(spec):
            raise BudgetExceeded(
                f"加载预算不足：{spec.name} 需要 weight={spec.weight}，"
                f"剩余预算 {self.remaining}"
            )
        self.spent += spec.weight

    def to_dict(self) -> Dict[str, Any]:
        return {"capacity": self.capacity, "spent": self.spent, "remaining": self.remaining}


class SkillRuntime:
    """
    按需加载 skill 实现（L2）与资源（L3）。

    skill.py 的约定（两种写法任选其一）：

      1) 函数式：
         def run(ctx, text: str, steps: int = 5):      # 返回 dict 或生成器
             for i in range(steps):
                 yield Progress(done=i + 1, total=steps, message="...")
             return {"count": len(text.split())}

      2) 类式：
         class Skill:
             def run(self, ctx, **inputs): ...

    生成器里 yield 的非 Progress 值 = 提前给出最终输出；return 值 = 最终输出。
    ctx 是 engine 提供的 StageContext（见 engine.py）。
    """

    def __init__(self, budget: Optional[LoadBudget] = None):
        self.budget = budget or LoadBudget(capacity=10**6)
        self._impls: Dict[str, Callable[..., Any]] = {}
        self._impl_mtimes: Dict[str, int] = {}
        self._resources: Dict[str, Dict[str, bytes]] = {}  # name -> {resname: bytes}

    # ── L2：实现懒加载 ────────────────────────────────────────

    def get_impl(self, spec: SkillSpec) -> Callable[..., Any]:
        """获取 skill 的可调用实现；必要时（首次/文件变更）重新加载。"""
        impl_path = spec.impl_file
        if impl_path is None or not impl_path.exists():
            raise FileNotFoundError(f"{spec.name} 缺少实现文件 skill.py")

        mtime = impl_path.stat().st_mtime_ns
        cached = self._impls.get(spec.name)
        if cached is not None and self._impl_mtimes.get(spec.name) == mtime:
            return cached

        self.budget.spend(spec)  # 预算不足会抛 BudgetExceeded（推迟执行）
        callable_impl = self._import_impl(spec)
        self._impls[spec.name] = callable_impl
        self._impl_mtimes[spec.name] = mtime
        return callable_impl

    @staticmethod
    def _import_impl(spec: SkillSpec) -> Callable[..., Any]:
        impl_path = spec.impl_file
        # 用唯一模块名加载，避免热更新时 sys.modules 里的旧模块互相覆盖
        module_name = f"_skillflow_{spec.name}_{int(time.time() * 1000)}"
        spec_ = importlib.util.spec_from_file_location(module_name, impl_path)
        if spec_ is None or spec_.loader is None:
            raise ImportError(f"无法加载 {impl_path}")
        module = importlib.util.module_from_spec(spec_)
        sys.modules[module_name] = module
        spec_.loader.exec_module(module)

        if hasattr(module, "run"):
            return module.run
        if hasattr(module, "Skill"):
            skill_cls = module.Skill
            return lambda ctx, **kw: skill_cls().run(ctx, **kw)
        raise AttributeError(f"{spec.name}/skill.py 需要定义 run() 或 Skill 类")

    def is_loaded(self, name: str) -> bool:
        return name in self._impls

    def unload(self, name: str):
        """卸载实现（释放预算），下次执行时重新加载。"""
        self._impls.pop(name, None)
        self._impl_mtimes.pop(name, None)

    def loaded_names(self) -> List[str]:
        return sorted(self._impls.keys())

    # ── L3：资源懒加载 ────────────────────────────────────────

    def list_resources(self, spec: SkillSpec) -> List[Dict[str, Any]]:
        """只列资源清单（文件名/大小），不读内容 —— 发现资源的成本是 O(1)。"""
        out = []
        rd = spec.resources_dir
        if rd is not None and rd.exists():
            for p in sorted(rd.iterdir()):
                if p.is_file():
                    out.append({"name": p.name, "size": p.stat().st_size})
        return out

    def load_resource(self, spec: SkillSpec, name: str) -> bytes:
        """按需读入单个资源并缓存。"""
        rd = spec.resources_dir
        if rd is None:
            raise FileNotFoundError(f"{spec.name} 没有 resources/ 目录")
        cached = self._resources.get(spec.name, {}).get(name)
        if cached is not None:
            return cached
        path = rd / name
        if not path.exists():
            raise FileNotFoundError(f"{spec.name} 缺少资源: {name}")
        data = path.read_bytes()
        self._resources.setdefault(spec.name, {})[name] = data
        return data

    def load_resource_text(self, spec: SkillSpec, name: str, encoding: str = "utf-8") -> str:
        return self.load_resource(spec, name).decode(encoding)
