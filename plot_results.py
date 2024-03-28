import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle as pck

path = "./results/"
names = sys.argv[1].split("-")

score_names = ['Rand', 'MI', 'Comp', 'FM']
f = open('results/' + sys.argv[1] + '_' + '_scores.tex', 'w')
f.write('\\begin{tabular}{| c | c | c  c  c | c  c |} \n')
f.write('\\hline \n')
f.write(' & & PCA & t-SNE & UMAP & Mapper (base) & Mapper (optim) \\\\ \n')
f.write('\\hline \n')
for name in names:
    results = pck.load(open(path + name + "/results.pkl", "rb"))
    scores_baseline = results['scores_baseline']
    scores = results['scores']
#    scores_baseline = [[s] for s in scores_baseline]
#    scores = [[s] for s in scores]
    for idx_s, score_name in enumerate(score_names):
        if idx_s == 0:
            f.write(name)
        scores_to_compare = [scores_baseline[i][idx_s] for i in range(3)] + [scores[i][idx_s] for i in [0,5]]
        idx_best = np.argmax(scores_to_compare)
        scores_to_write = ['{0:.3g}'.format(sc) for sc in scores_to_compare]
        scores_to_write[idx_best] = '\\bf{' + scores_to_write[idx_best] + '}'
        f.write(' & ' + score_name + ' & ')
        f.write(' & '.join(scores_to_write) + ' \\\\ \n')
    f.write('\\hline \n')
f.write('\\end{tabular}')
f.close()

score_names = ['Rand', 'MI', 'Comp', 'FM']
f = open('results/' + sys.argv[1] + '_' + '_costs.tex', 'w')
f.write('\\begin{tabular}{| c | c  c  c | c  c |} \n')
f.write('\\hline \n')
f.write(' & PCA & t-SNE & UMAP & Mapper (base) & Mapper (optim) \\\\ \n')
f.write('\\hline \n')
for name in names:
    results = pck.load(open(path + name + "/results.pkl", "rb"))
    costs_baseline = results['costs_baseline']
    costs = results['costs']
    f.write(name + ' & ' + '{0:.3g}'.format(costs_baseline[0]) + ' & ' + '{0:.3g}'.format(costs_baseline[1]) + ' & ' + '{0:.3g}'.format(costs_baseline[2]) + ' & ')
    f.write(               '{0:.3g}'.format(costs[0]) + ' & ' + '{0:.3g}'.format(costs[5]) + ' \\\\ \n')
    f.write('\\hline \n')
f.write('\\end{tabular}')
f.close()
