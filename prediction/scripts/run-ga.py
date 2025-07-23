# python -m prediction.scripts.run-ga.py

from prediction import genetics

best_genome = genetics.search(obliterate_cache=True)