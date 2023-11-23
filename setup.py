# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='idpatregpy',
    version='0.0.1',
    description='Pattern recognition system for Xenopus dorsal pattern identification',
    long_description=readme,
    author='Pubordee Aussavavirojekul',
    author_email='pubordee.a@gmail.com',
    url='https://github.com/eedrobup/idpatregpy.git',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires = required,
    extras_require={
        "dev": ["pytest>=7.0","twine>=4.0.2"]
    },
    python_requires=">=3.10"
)