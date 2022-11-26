if [ ! -d "./.venv" ] 
then
    echo "### Creating virtual environment ###"
    python -m venv .venv
    source ./.venv/bin/activate

    echo "### Installing dependencies ###"
    python -m pip install --upgrade pip
    python -m pip install -r requirements_dev.txt
    python -m pip install -r requirements.txt
    python -m pip install .
else
    source ./.venv/bin/activate
fi

echo "### Writing RAM and GPU info"
nvidia-smi
free -h

echo "### Starting script ###"
python hpo.py --dataset_directory 12_8_0_12_1.0