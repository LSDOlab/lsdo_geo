# Getting Started

This page provides instructions for installing **lsdo_geo**, running the test suite, and building the documentation.

* **GitHub Repository**: [github.com/afletche/lsdo_geo](https://github.com/afletche/lsdo_geo)
* **LSDO Lab**: [lsdo.eng.ucsd.edu](https://lsdo.eng.ucsd.edu/)
* **CSDL Framework**: [github.com/LSDOlab/csdl](https://github.com/LSDOlab/csdl)

---

## Installation

### For Users
To install `lsdo_geo` directly from GitHub:
```sh
pip install git+https://github.com/afletche/lsdo_geo.git
```

### For Developers
Clone the repository and install it in editable mode with development dependencies:
```sh
git clone https://github.com/afletche/lsdo_geo.git
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
