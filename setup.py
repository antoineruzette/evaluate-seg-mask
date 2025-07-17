from setuptools import setup, find_packages
import os

setup(
    name="evaluatesegmask",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': ['data/**/*'],
    },
    install_requires=[
        "numpy",
        "scikit-image",
        "scipy",
        "pandas",
        "requests",
        "Pillow"
    ],
    author="Antoine",
    author_email="",  # Add your email if desired
    description="A package for evaluating instance segmentation masks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",  # Add your repository URL if desired
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "evaluatesegmask=evaluatesegmask.cli:main",
        ],
    },
) 