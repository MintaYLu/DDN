# DDN 3.0
We developed an efficient and accurate differential network analysis tool – Differential Dependency Networks (DDN3.0).

DDN 3.0 is capable of jointly learning sparse common and rewired network structures, which is especially useful for genomics, proteomics, and other biomedical studies.

This repository provides the Python package and examples of using DDN 3.0.

## Installation
DDN 3.0 is still in development and is not in PyPI yet.
It is recommended to install it inside a Conda environment.

First we need to install some common dependencies.
```bash
$ conda install -c conda-forge numpy scipy numba networkx matplotlib jupyter scipy pandas scikit-learn
```

Clone the repository, or just download or unzip it.

Then we can install DDN 3.0 in developement mode.
```bash
$ pip install -e ./
```

## Usage

This toy example generates two random datasets, and use estimate to estimate two networks, one for each dataset.
```python
import numpy as np
from ddn import ddn
dat1 = np.random.randn(1000, 10)
dat2 = np.random.randn(1000, 10)
networks = ddn.ddn(dat1, dat2, lambda1=0.3, lambda2=0.1)
```

For more details and examples, check the three tutorials in the `notebooks` folder.

## Contributing

Please report bugs in the issues. 
You may also email the authors directly: Yingzhou Lu (lyz66@vt.edu), Yizhi Wang (yzwang@vt.edu), or Yue Wang (yuewang@vt.edu).
If you are interested in adding features or fixing bug, feel free to contact us.

## License

The `ddn` package is licensed under the terms of the MIT license.

## Citations

[1] Zhang, Bai, and Yue Wang. "Learning structural changes of Gaussian graphical models in controlled experiments." arXiv preprint arXiv:1203.3532 (2012).
