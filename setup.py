try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

try:
    import builtins
except ImportError:
    # Python 2 compat: just to be able to declare that Python >=3.5 is needed.
    import __builtin__ as builtins

import fsmpy

VERSION = fsmpy.__version__

CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development",
    "Topic :: Utilities",
    "Operating System :: Microsoft",
    "Operating System :: MacOS",
    "Operating System :: POSIX :: Linux",
]

# This call to setup() does all the work
setup(
    name="fuzzy-set-measures",
    version=VERSION,
    description="Generalized measures for application in fuzzy set theory.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    author="Machine Learning & Vision Research Group",
    license="BSD 3-Clause License",
    classifiers=CLASSIFIERS,
    packages=[
        "fsmpy", "fsmpy.utils"
    ],
    include_package_data=True,
    install_requires=[
        "numpy >= 1.14.6",
        "scikit-learn >=0.24.2",
        "tqdm >= 4.62.1",
    ],
)