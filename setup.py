from setuptools import setup, find_packages

with open("esm/version.py") as f:
    exec(f.read())

setup(
    name="deepRAD51C",
    version=version,  # noqa: F821  — set by exec above
    description="WT-difference embedding model for RAD51C variant effect prediction",
    author="",
    python_requires=">=3.8",
    packages=find_packages(exclude=["src*", "_archive_removed*"]),
    install_requires=[
        "torch>=1.12",
        "fair-esm",
    ],
)
