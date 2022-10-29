# nowcasting

## Description


## Setup
Create a virtual environment to use as an isolated workspace

```bash
python -m pip install --upgrade virtualenv
python -m venv .venv
```

Activate your virtual environment with the command appropriate for your operating system

Linux
```bash
source ./.venv/bin/activate
```

Windows
```powershell
./.venv/Scripts/activate
```

Update your pip install and build packages. Install the required packages including a developer install of the code in ./src. Use the developer install while in the root directory of this project.

```bash
python -m pip install --upgrade pip
python -m pip install --upgrade build wheel setuptools

python -m pip install -r requirements_dev.txt
python -m pip install -r requirements.txt
python -m pip install -e .
```


## Unit tests
Install several python testing packages and run ```pytest``` using the "tests" directory as an argument.

```bash
python -m pip install pytest pytest-cov
python -m pytest tests
```

