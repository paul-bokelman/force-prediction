# python -m prediction.scripts.run-ga.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all tf except fatal errors

from prediction import genetics
best_genome = genetics.search(obliterate_cache=True)