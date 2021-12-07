# Autodetect
A package for a score-based change detection method, *auto-test*, that can automatically report hidden changes in machine learning systems as they learn from a continuous, possibly evolving, stream of data.
This is code accompanying the paper "[Score-Based Change Detection for Gradient-Based Learning Machines](https://arxiv.org/abs/2106.14122)" in ICASSP 2021.

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
@inproceedings{lsh2021,
title = {Score-Based Change Detection for Gradient-Based Learning Machines},
author = {Liu, Lang and
          Salmon, Joseph and
          Harchaoui, Zaid},
booktitle = {2021 {IEEE} International Conference on Acoustics, Speech and Signal Processing, {ICASSP} 2021, Toronto, Canada, June 6-11, 2021},
publisher = {{IEEE}},
year = {2021}
}
```

## Acknowledgements
This work was supported by NSF CCF-1740551, NSF DMS-1810975, the program “Learning in
Machines and Brains” of CIFAR, and faculty research awards.
