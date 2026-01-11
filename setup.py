"""Setup script for LineamentLearning package."""

from setuptools import setup, find_packages
import os

# Read the README file
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name='lineament-learning',
    version='2.0.0',
    author='Amin Aghaee',
    description='Deep Learning for Lineament Detection in Geoscience Data',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/RichardScottOZ/LineamentLearning',
    packages=find_packages(exclude=['tests', 'examples']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0,<2.0.0',
        'scipy>=1.7.0',
        'pillow>=9.0.0',
        'tensorflow>=2.10.0,<2.16.0',
        'keras>=2.10.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.5.0',
        'h5py>=3.7.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'modern-ui': [
            'gradio>=3.0.0',
            'streamlit>=1.20.0',
        ],
        'full': [
            'tensorboard>=2.10.0',
            'pyyaml>=6.0',
            'click>=8.0.0',
            'tqdm>=4.64.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'lineament-train=cli:train_command',
            'lineament-predict=cli:predict_command',
        ],
    },
)
