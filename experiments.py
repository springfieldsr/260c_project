# LRS = [1e-2,1e-3,1e-4,1e-5]
LRS = [1e-4]
LOSS_PROCESS_METHODS = ['Std','Mean','Dist']
# TOP_KS = [0.05,0.1,0.2,0.4]
TOP_KS = [0.05]

# GTOP_KS = [0.05,0.1,0.2,0.4,0.6]
GTOP_KS = [0.05,0.1]

import itertools

import os
def main():
    entries = os.path.join('.','NoisyDetection.py')
    for lr,lpm, tk,gk in itertools.product(LRS,LOSS_PROCESS_METHODS,TOP_KS,GTOP_KS):
        os.system('python {} --lr {} --k {} --gk {} --lp {}'.format(entries, lr,tk,gk,lpm))





if __name__ == '__main__':
    main()
