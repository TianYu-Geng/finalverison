# 同级目录里有一个 diffuser/ 文件夹（且里面有 __init__.py），
# 再通过一次安装命令（pip install -e .），把它注册到 Python 的包系统里，于是它就“成了包”。

from setuptools import setup, find_packages

setup(
  name = 'diffuser',
  packages = find_packages(), # 找到diffuser下面的所有包
)
