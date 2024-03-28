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
    for idx_s, score_name in enumerate(score_names):
        if idx_s == 0:
            f.write(name)
        scores_to_compare = []
        for run in range(1,2):
            results = pck.load(open(path + name + "_run" + str(run) + "/results.pkl", "rb"))
            scores_baseline = results['scores_baseline']
            scores = results['scores']
            scores_to_compare.append([scores_baseline[i][idx_s] for i in range(3)] + [scores[i][idx_s] for i in [0,5]])
        mean_scores, var_scores = np.mean(np.array(scores_to_compare), axis=0), np.var(np.array(scores_to_compare), axis=0)
        idx_best = np.argmax(mean_scores)
        scores_to_write = ['{0:.3g}'.format(mean_scores[idx_sc]) + ' $\\pm$ ' + '{0:.3g}'.format(var_scores[idx_sc]) for idx_sc in range(len(mean_scores))]
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
    f.write(name)
    costs_to_compare = []
    for run in range(1,2):
        results = pck.load(open(path + name + "_run" + str(run) + "/results.pkl", "rb"))
        costs_baseline = results['costs_baseline']
        costs = results['costs']
        costs_to_compare.append([costs_baseline[0], costs_baseline[1], costs_baseline[2], costs[0], costs[5]])
    mean_costs, var_costs = np.mean(np.array(costs_to_compare), axis=0), np.var(np.array(costs_to_compare), axis=0)
    idx_best = np.argmax(mean_costs)
    costs_to_write = ['{0:.3g}'.format(mean_costs[idx_co]) + ' $\\pm$ ' + '{0:.3g}'.format(var_costs[idx_co]) for idx_co in range(len(mean_costs))]
    costs_to_write[idx_best] = '\\bf{' + costs_to_write[idx_best] + '}'
    f.write(' & ')
    f.write(' & '.join(costs_to_write) + ' \\\\ \n')
    f.write('\\hline \n')
f.write('\\end{tabular}')
f.close()
