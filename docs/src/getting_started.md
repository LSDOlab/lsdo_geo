# Getting Started

This page provides instructions for installing **lsdo_geo**, running the test suite, and building the documentation.

## Installation

### For Users
To install `lsdo_geo` directly from GitHub:
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

## Running Tests
Run the test suite using pytest from the repository root:
```sh
pytest
```

To run only fast unit tests (excluding slow optimization benchmarks):
```sh
pytest -m "not slow"
```

## Building Documentation
To build the documentation locally with Sphinx:
```sh
cd docs
make html
```
The compiled HTML documentation will be generated in `docs/_build/html/index.html`.
