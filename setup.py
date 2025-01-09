import sys
from setuptools import setup, find_packages

sys.path[0:0] = ["plmbind"]
from version import __version__

setup(
    name="plmbind",
    python_requires=">3.9.0",
    packages=find_packages(),
    version=__version__,
    license="MIT",
    description="TF binding prediction using protein language models (pLMs)",
    author="Natan Tourné",
    url="https://github.com/natantourne/thesis",
    install_requires=[
        "numpy",
        "torch",
        "pytorch-lightning",
        "h5torch"
    ],
)