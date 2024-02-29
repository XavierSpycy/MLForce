from setuptools import setup, find_packages
import mlforce

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

VERSION = mlforce.__version__

setup(
    name="mlforce",
    version=VERSION,
    author="Jiarui Xu",
    author_email="xujiarui98@foxmail.com",
    description="Easy-to-use machine learning toolkit for beginners",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XavierSpycy/MLForce",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "joblib",
    ],
)