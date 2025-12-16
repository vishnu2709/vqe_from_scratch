import numpy as np
from vqe import do_full_vqe

dist = np.linspace(0.2, 6, 30)
#dist = [0.74]
energy = []
starting_guess = (0.5, 0.5, 0.5)
for d in dist:
   e, new_guess = do_full_vqe(d, starting_guess)
   print(new_guess)
   #starting_guess = new_guess
   energy.append(e)
  

for i in range(len(dist)):
   print(dist[i], energy[i].real)
