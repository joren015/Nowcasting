echo "### Installing dependencies ###"

pip install -r requirements_dev.txt
pip install -r requirements.txt
pip install --upgrade scipy
pip install .

echo "### Writing RAM and GPU info"
nvidia-smi
free -h

echo "### Starting GPU monitor ###"
ts=$(date +%s)
free -h >> ./data/ram/ram_stats_$ts.txt &
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f ./data/gpu/gpu_stats_$ts.csv &

echo "### Starting script ###"

python hpo.py