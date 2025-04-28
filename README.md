# FastShermanMorrison

![PyPI](https://img.shields.io/pypi/v/fastshermanmorrison-pulsar)
![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/fastshermanmorrison-pulsar)


Cython code to more quickly evaluate ShermanMorrison combinations as need by
kernel ecorr in Enterprise.

# Installation

The FastShermanMorrison add-on to Enterprise can be easily installed straight
from github using

```bash
pip install git+https://github.com/nanograv/fastshermanmorrison.git
```

From Pypi, you can do

```bash
pip install fastshermanmorrison-pulsar
```

Conda support is in testing stage. Apple silicon arm processors are not supported yet, but on other architectures you can do

```
conda install -c vhaasteren fastshermanmorrison-pulsar
```

Availability on conda-forge is upcoming in a later release

## citation
Did you use `fastshermanmorrison`?
Remember to cite it as:

>van Haasteren, Rutger (2023). fastshermanmorrison v0.5.2. https://github.com/nanograv/fastshermanmorrison

```latex
@misc{fastshermanmorrison,
  author       = {Rutger van Haasteren},
  title        = {fastshermanmorrison},
  year         = {2023},
  url          = {https://github.com/nanograv/fastshermanmorrison},
  note         = {v0.5.2}
}
```

