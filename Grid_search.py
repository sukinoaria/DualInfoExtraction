import os

iteration = 3

for h2bh in range(5,15,2):
    for h2bb in range(5, 15, 2):
        for b2hb in range(5, 15, 2):
            for b2hh in range(5, 15, 2):
                os.system('python main.py --iteration {} --H2BH {} --H2BB {} --B2HB {} --B2HH {}'.format(iteration,h2bh,h2bb,b2hb,b2hh))
