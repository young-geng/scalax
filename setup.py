from setuptools import setup, find_packages

setup(
    name='scalax',
    version='0.1.3',
    license='Apache-2.0',
    description='Scaling utilities for JAX.',
    url='https://github.com/young-geng/scalax',
    packages=find_packages(include=['scalax']),
    python_requires=">=3.8",
    install_requires=[
        'numpy',
        'jax',
        'einops',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
    ],
)