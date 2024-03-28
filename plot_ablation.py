import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle as pck

path = "./results/"
names = sys.argv[1].split("-")

score_names = ['Rand', 'MI', 'Comp', 'FM']

f = open('results/' + sys.argv[1] + '_' + '_corrs.tex', 'w')
f.write('\\begin{tabular}{| c | ' + ' '.join(['c |' for _ in range(len(names))]) + '} \n')
f.write('\\hline \n')
f.write(' & ' + ' & '.join(names) + ' \\\\ \n')
f.write('\\hline \n')
f.write('Corr. & ')
corr_vals = []
for name in names:
    res = pck.load(open(path + name + "/results.pkl", "rb"))
    corr_vals.append(res['corr_mapper'][1][0])
corrs = ['{0:.3g}'.format(c) for c in corr_vals]
f.write(' & '.join(corrs) + ' \\\\ \n')
f.write('\\hline \n')
for idx_s, score_name in enumerate(score_names):
    f.write(score_name + ' & ')
    scores_to_compare = []
    for name in names:
        res = pck.load(open(path + name + "/results.pkl", "rb"))
        scores = res['scores']
        scores_to_compare.append(scores[5][idx_s])
    scores_to_write = ['{0:.3g}'.format(sc) for sc in scores_to_compare]
    f.write(' & '.join(scores_to_write) + ' \\\\ \n')
    f.write('\\hline \n')
f.write('\\end{tabular}')
f.close() 

