import catalogue
import numpy as np

true_effects = catalogue.create("power", "true_effects")

@effects.register("effect::cohens_g")
