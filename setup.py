from setuptools import setup, find_packages
import os

# setup.py is inside Anytop/, so we look one level up for the "Anytop" package
setup_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(setup_dir)

setup(
    name="Anytop",
    version="0.1.0",
    package_dir={"Anytop": "."},
    packages=["Anytop"] + [p for p in find_packages(where=".") if p != "__pycache__"],
    include_package_data=True,
)
