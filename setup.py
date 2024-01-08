from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlforce",
    version="1.0.0",
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
    ],
)