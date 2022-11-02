
jupyter nbconvert --ExecutePreprocessor.timeout=43200 --to notebook --execute notebooks/hpo.ipynb > hpo_output.txt 2>&1

