import sys
from setuptools import setup, find_packages

sys.path[0:0] = ["TFBS_negatives"]
#from version import __version__

setup(
    name="TFBS_negatives",
    python_requires=">3.9.0",
    packages=find_packages(),
    #version=__version__,
    license="MIT",
    description="TF binding prediction",
    author="Natan Tourné",
    url="",
    install_requires=[
        "numpy",
        "torch",
        "pytorch-lightning",
        "h5torch"
    ],
)