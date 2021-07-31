# FunnkSVD
FunkSVD is an effective algorithm developed as part of the Netflix Prize competition and described by the author [in this blogpost](https://sifter.org/simon/journal/20061211.html)

We used a Python3 implementation of the algorithm available in [GitHub](https://github.com/gbolmier/funk-svd) to calculate this baseline.

## Requirements and commands

The requirements needed to reproduce the baseline results are contained in the file ``FunkSVD_requirements.txt``
This code was tested when using Python 3.8.x

To install the funk-svd python library run ```pip install git+https://github.com/gbolmier/funk-svd```

To train, evaluate the model run ```python FunkSVD.py``` in this subfolder.
