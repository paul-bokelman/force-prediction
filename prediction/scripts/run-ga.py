from prediction import genetics
best_genome = genetics.search(obliterate_cache=True)
# metrics.generate_report(best_genome.to_model_candidate())