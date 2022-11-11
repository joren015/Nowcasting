echo "### Installing dependencies ###"

pip install -r requirements_dev.txt
pip install -r requirements.txt
pip install .

echo "### Starting GPU monitor ###"
ts=$(date +%s)
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f ./data/gpu/gpu_stats_$ts.csv &

echo "### Starting script ###"

python hpo.py