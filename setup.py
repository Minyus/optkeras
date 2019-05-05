from setuptools import setup
from codecs import open
from os import path
import re

package_name = 'optkeras'

# Read version from __init__.py file
root_dir = path.abspath(path.dirname(__file__))
with open(path.join(root_dir, package_name, '__init__.py')) as f:
    init_text = f.read()
    version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)

setup(
    name=package_name,
    packages=[package_name],
    version=version,
    license='MIT',
    author='Yusuke Minami',
    author_email='me@minyus.github.com',
    url='https://github.com/Minyus/optkeras',
	description='OptKeras: Wrapper of Keras and Optuna to optimize hyperparameters of Deep Learning.',
	install_requires=[
        'keras',
        'optuna==0.7.0',
        'numpy'
    ],
	keywords='keras optuna',
	zip_safe=False,
    test_suite='tests',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3.6',
        "Operating System :: OS Independent"
    ])
