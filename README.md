# Autodetect
A package for a score-based change detection method, *autograd-test*, that can automatically report hidden changes in machine learning systems as they learn from a continuous, possibly evolving, stream of data.
This is code accompanying the paper "[Score-based Change Detection for Gradient-based Learning Machines](https://stat.uw.edu/sites/default/files/2019-07/tr652.pdf)".

## Prerequisites
This package is based on [PyTorch](https://pytorch.org/). Other dependencies can be found in the file [environment.yml](environment.yml).
If using ``conda``, run the following command to install all required packages and activate the environment:
```bash
$ conda env create --file environment.yml
$ source activate autodetect
```

## Installation
Clone the repository here:
```bash
$ git clone https://github.com/langliu95/autodetect.git
$ cd autodetect/
```

## Documentation
The documentation for this package can be found [here](https://www.stat.washington.edu/~liu16/autodetect/).

## Authors
* Lang Liu
* [Joseph Salmon](http://josephsalmon.eu/)
* [Zaid Harchaoui](http://faculty.washington.edu/zaid/)

## License
This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

## Cite
If you use this code, please cite:

```
@techreport{lsh2019,
title = {Score-based Change Detection for Gradient-based Learning Machines},
author = {Liu, Lang and
          Salmon, Joseph and
          Harchaoui, Zaid},
year = {2019},
institution = {Department of Statistics, University of Washington},
month = {June}
}
```

## Acknowledgements
This work was supported by NSF CCF-1740551, NSF DMS-1810975, the program “Learning in
Machines and Brains” of CIFAR, and faculty research awards.