from setuptools import setup, find_packages

setup(
    name="instruction_generation",
    version="0.1",
    packages=find_packages(where="instruction_generation_pkg"),
    package_dir={"": "instruction_generation_pkg"},
)
