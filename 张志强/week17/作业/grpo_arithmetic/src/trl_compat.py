"""
trl 0.21 + transformers 5.x 兼容性修复（必须在 import trl 训练器之前执行）

问题根因：
  trl 0.21 的 import_utils.py 使用了 transformers 的私有函数 _is_package_available()，
  在 transformers 4.x 中它返回 bool，而 transformers 5.x 改为始终返回 (bool, version) 元组。
  Python 中非空元组恒为 truthy，导致 trl 把所有可选项包（vllm/deepspeed/unsloth 等）
  都误判为"已安装"，import GRPOTrainer 时顶层执行 `import vllm` 直接崩溃。

修复思路：
  在 trl 的各子模块被导入之前，把 trl.import_utils 里所有返回元组的
  is_*_available() 函数替换为返回真正的 bool。trl 的懒加载机制保证
  各 trainer 模块在 from trl import XXX 时才导入，因此此处补丁先生效。

用法（所有使用 trl 的脚本，第一行先 import 本模块）：
  import trl_compat  # noqa: F401  必须先于 trl 导入
  from trl import GRPOTrainer, GRPOConfig
"""


def _patch_trl_availability_flags():
    import trl.import_utils as tiu

    for name in dir(tiu):
        if not (name.startswith("is_") and name.endswith("_available")):
            continue
        fn = getattr(tiu, name)
        if not callable(fn):
            continue
        try:
            val = fn()
        except Exception:
            continue
        if isinstance(val, tuple):  # transformers 5.x 的元组返回值 → 取真实布尔位
            real = bool(val[0])
            setattr(tiu, name, lambda *a, _r=real, **k: _r)


def _patch_disable_mps():
    """macOS 26.x + torch 2.13.0 的 MPS 后端处于"编译了但运行时不可用"的损坏状态：
    _has_mps=True 但 _mps_is_available()=False。部分代码路径（accelerate 设备检测、
    DataLoader pin_memory 线程）仅检查 _has_mps 就尝试 .to('mps')，触发
    MetalShaderLibrary::getFunctionNames() 空指针解引用 → segfault。
    强制设为 False 可让所有设备检测逻辑回退到 CPU。"""
    import torch._C as C

    if hasattr(C, "_has_mps"):
        C._has_mps = False


def _patch_warnings_issued():
    """transformers 5.x 移除了 PreTrainedModel.warnings_issued（4.x 的警告去重字典），
    trl 0.21 的 GRPOTrainer 初始化时仍直接读写它。补一个类级别的空字典即可——
    trl 只是往里面写标记位，不影响训练逻辑。"""
    from transformers import PreTrainedModel

    if not hasattr(PreTrainedModel, "warnings_issued"):
        PreTrainedModel.warnings_issued = {}


_patch_trl_availability_flags()
_patch_disable_mps()
_patch_warnings_issued()
