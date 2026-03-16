# TMA4180-Project-1

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── .gitignore
├── src/
│   └── nummat_project2/
│       ├── __init__.py
│       ├── main.py
│       ├── fields.py
│       ├── solvers.py
│       └── plotting.py
└── notebooks/                   
    ├── main.ipynb
    └── notebook.ipynb
```

## Installation

Create a virtual environment and install the project.

### Create environment

Windows

```
py -3.12 -m venv .venv
```

Mac / Linux

```
python -m venv .venv
```

Activate environment

Windows

```
.venv\Scripts\activate
```

Mac / Linux

```
source .venv/bin/activate
```

Install the project

```
pip install -e .
```

---