# Getting Started

This page provides instructions for installing **lsdo_geo**, running the test suite, and building the documentation.

* **GitHub Repository**: [github.com/LSDOlab/lsdo_geo](https://github.com/LSDOlab/lsdo_geo)
* **LSDO Lab**: [lsdo.eng.ucsd.edu](https://lsdo.eng.ucsd.edu/)
* **CSDL Framework (CSDL Alpha)**: [github.com/LSDOlab/CSDL_alpha](https://github.com/LSDOlab/CSDL_alpha)

---

## Installation

### Prerequisites
`lsdo_geo` relies on `CSDL_alpha` and `lsdo_function_spaces`:

```sh
pip install jax
pip install git+https://github.com/LSDOlab/CSDL_alpha.git
pip install git+https://github.com/LSDOlab/lsdo_function_spaces.git
```

### For Users
Install `lsdo_geo` via pip:
```sh
pip install lsdo_geo
```
Or install the latest development version directly from GitHub:
```sh
pip install git+https://github.com/LSDOlab/lsdo_geo.git
```

### For Developers
Clone the repository and install it in editable mode with development dependencies:
```sh
git clone https://github.com/LSDOlab/lsdo_geo.git
cd lsdo_geo
pip install -e ".[test,docs]"
```

---

## Running Tests
Run the test suite using pytest from the repository root:
```sh
pytest
```

To run only fast unit tests (excluding slow optimization benchmarks):
```sh
pytest -m "not slow"
```

---

## Building Documentation
To build the documentation locally with Sphinx:
```sh
cd docs
make html
```
The compiled HTML documentation will be generated in `docs/_build/html/index.html`.

---

## Next Steps
* Explore the [Examples](examples.md) to see wing parameterizations and MDO workflows.
* Check out the [Interactive Tutorials](tutorials.md).
* Inspect the automatically generated [API Reference](api.md).
