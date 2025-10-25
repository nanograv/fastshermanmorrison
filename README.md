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

## Citation

If you use `fastshermanmorrison-pulsar` in your research, please cite it as follows:

### BibTeX Entry
```bibtex
@software{fastshermanmorrison-pulsar,
  author       = {Rutger van Haasteren},
  title        = {fastshermanmorrison-pulsar: Fast Sherman-Morrison Updates for Pulsar Timing},
  year         = {2023},
  version      = {0.5.3},
  publisher    = {GitHub},
  url          = {https://github.com/nanograv/fastshermanmorrison},
  doi          = {10.5281/zenodo.17439722},
  note         = {Software for efficient Sherman-Morrison matrix updates in pulsar timing analysis}
}
```

### Text Citation
van Haasteren, R. (2023). fastshermanmorrison-pulsar: Fast Sherman-Morrison Updates for Pulsar Timing (Version 0.5.3) [Software]. GitHub. https://github.com/nanograv/fastshermanmorrison

### Alternative Citation (if DOI not available)
```bibtex
@manual{fastshermanmorrison-pulsar,
  title        = {fastshermanmorrison-pulsar: Fast Sherman-Morrison Updates for Pulsar Timing},
  author       = {Rutger van Haasteren},
  year         = {2023},
  note         = {Version 0.5.3},
  url          = {https://github.com/nanograv/fastshermanmorrison}
}
```

