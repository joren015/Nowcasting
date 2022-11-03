echo "### Installing dependencies ###"

pip install -r requirements_dev.txt
pip install -r requirements.txt
pip install .

echo "### Starting script ###"

python hpo.py