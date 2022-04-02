[![pypi](https://img.shields.io/pypi/v/fuzzy-set-measures.svg)](https://pypi.org/project/fuzzy-set-measures/)
# fsmpy Development Repository

![fsmpy_library_process](media/Library%20process.png)

fsmpy (Fuzzy Set Measures) is a Python module for the application of Fuzzy Set Theory and is distributed under the 3-Clause BSD license.

website: https://machinelearningvisionrg.github.io/fsmpy-docs/

# Installation

## Dependencies

* Python (>=3.7)
* NumPy (>= 1.14.6)
* scikit-learn (>=0.24.2)

## User installation

If you have a working installation of NumPy and scikit-learn, the simplest way to install fsmpy is using the package installer for Python, **pip**
```
pip install fuzzy-set-measures
```
<!-- or the package management system **conda**
```
conda install fuzzy-set-measures
``` -->

# Changelog
See the changelog for a history important changes to the library.

# Development & Contributions

All contributions of any level and kind are welcome. Please follow the Development Guide for further information about the contribution process, documentation, tests and more. 

All tests are run by executing ``pytest`` in the top level directory.
No subset of tests is available for the time being.

## Source code
You can get the latest version of the source code using this command:
```
git clone https://github.com/MachineLearningVisionRG/fsmpy
```

## Pull request submission
Before opening a pull request, take a look at the [contribution](CONTRIBUTING.md) page.

# Examples
Some basic usage examples are provided below. Please take a look at the documentation for further information and detailed examples.
## Fuzzy sets representation
Fuzzy Sets are represented through the IntuitionisticFuzzySet class which includes attributes for the corresponding membership and non-membership values. A Fuzzy Set *S* with membership and non-membership values is initialized in the following manner:
```
S = IntuitionisticFuzzySet(membership_values: Iterable, non_membership_values: Iterable = None)
```


To represent the following Fuzzy Set patterns in ![X](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20X%20%3D%20%5C%7Bx_1%2C%20x_2%2C%20x_3%5C%7D):

![S1](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20S_1%3D%5C%7B%28x_1%2C%200.5%2C%200.4%7Cx_1%29%2C%20%28x_2%2C%200.8%2C%200.0%7Cx_2%29%2C%20%28x_3%2C%200.7%2C%200.1%7Cx_3%29%5C%7D)

![S2](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20S_2%3D%5C%7B%28x_1%2C%201.0%2C%200.0%7Cx_1%29%2C%20%28x_3%2C%201.0%2C%200.1%7Cx_3%29%5C%7D)

![S3](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20S_3%3D%5C%7B%28x_1%2C%200.9%2C%200.5%7Cx_1%29%2C%20%28x_2%2C%200.8%2C%200.3%7Cx_2%29%5C%7D)

use the IntuitionisticFuzzySet class to initialize an object like so:
```
S1 = IntuitionisticFuzzySet([0.5, 0.8, 0.7], [0.4, 0.0, 0.1])
S1 = IntuitionisticFuzzySet([1.0, 0.0, 1.0], [0.0, 0.0, 0.1])
S1 = IntuitionisticFuzzySet([0.9, 0.8, 0.0], [0.5, 0.3, 0.0])
```

Note that patterns that do not represent a set should be set to 0. 

## Fuzzy measure usage
Calculate the normalized Euclidean distance between two Fuzzy Sets A and B:
```
import fsmpy as fsm
from fsmpy.distances import atanassov

atanassov(A, B, fsm.DISTANCE_NORMALIZED_EUCLIDEAN)
```

Calculate the second similarity measure proposed by Liang and Shi (2003):
```
import fsmpy as fsm
from fsmpy.similarities import liang_shi

liang_shi(A, B, fsm.LIANG_SHI_SIMILARITY_2, p=2)
```

## Pattern Recognition
Load the provided medical diagnosis data used in the literature and classify the first patient's symptoms to the corresponding diagnosis with the distance measure proposed by Wang and Xin (2005), with $p=2$:
```
from fsmpy.distances import wang_xin
from fsmpy.utils import classify
from fsmpy.datasets import load_patients_diagnoses

diagnoses, patients = load_patients_diagnoses()
classify(diagnoses, patients[0], wang_xin, p=2)
```

# Citation
If you use fsmpy in a scientific publication, please use the following bibtex citation:
```
@article{
      2022, 
      title={Fsmpy: A Fuzzy Set Measures Python Library}, 
      volume={13}, 
      ISSN={2078-2489}, 
      url={http://dx.doi.org/10.3390/info13020064}, 
      DOI={10.3390/info13020064}, 
      number={2}, 
      journal={Information}, 
      publisher={MDPI AG}, 
      author={Sidiropoulos, George K. and Apostolidis, Kyriakos D. and Damianos, Nikolaos and Papakostas, George A.}, 
      year={2022}, 
      month={Jan}, 
      pages={64}
}
```
