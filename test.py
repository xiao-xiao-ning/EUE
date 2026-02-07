# 创建一个测试脚本 test_numpy.py
import numpy as np

print(f"NumPy版本: {np.__version__}")
print(f"安装路径: {np.__file__}")

# 检查trapz的各种可能位置
print("\n检查trapz位置:")
print(f"1. np.trapz 存在? {hasattr(np, 'trapz')}")
print(f"2. np.lib.trapz 存在? {hasattr(np.lib, 'trapz')}")

# 查看numpy的所有子模块
print("\nNumPy的子模块:")
import pkgutil
for importer, modname, ispkg in pkgutil.iter_modules(np.__path__):
    print(f"  {modname}")

# 检查numpy.lib的内容
print("\n检查numpy.lib模块:")
import numpy.lib
print(dir(numpy.lib)[-20:])  # 显示最后20个属性