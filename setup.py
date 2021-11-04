#!/usr/bin/env python
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="domain-adaptation",
    version='0.1',
    description="Domain adaptation framework for Transformers and other PyTorch networks.",
    long_description="Framework for a textual domain adaptation, applicable for any PyTorch network.",
    classifiers=[],
    author="To Be Added",
    author_email="tobeadded@tobeadded.com",
    url="gitlab.com",
    license="MIT",
    packages=find_packages(include=["domain_adaptation"]),
    include_package_data=True,
    zip_safe=True,
    install_requires=[
        "torch>=1.7",
        "transformers[sentencepiece]==4.10.2",
        "torch",
        "pytest",
        "sacrebleu"
    ],
)
