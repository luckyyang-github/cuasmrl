from setuptools import setup

setup(
    name="cuasmrl",
    version="0.0.1",
    install_requires=[
        # 进度条库，verify.py 等模块依赖
        "tqdm>=4.60",
    ],
)
