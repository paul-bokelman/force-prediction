import random

def random_exclude(*exclude, min: int, max: int):
  """Generate a random integer between 0 and 9, excluding specified values."""
  exclude = set(exclude)
  randInt = random.randint(min,max)
  return random_exclude(*exclude, min=min, max=max) if randInt in exclude else randInt 

def chance(probability: float):
  """Determine if an event occurs based on a given probability."""
  return random.randint(1, 10) <= probability * 10