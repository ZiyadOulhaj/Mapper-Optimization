import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle as pck

path = "./results/"
names = sys.argv[1].split("-")

score_names = ['Rand', 'MI', 'Comp', 'FM']
num_runs = 1

f = open('results/' + sys.argv[1] + '_' + '_corrs.tex', 'w')
f.write('\\begin{tabular}{| c | ' + ' '.join(['c |' for _ in range(len(names))]) + '} \n')
f.write('\\hline \n')
f.write(' & ' + ' & '.join(names) + ' \\\\ \n')
f.write('\\hline \n')
f.write('Corr. & ')
corr_means, corr_vars = [], []
for name in names:
    corr_runs = []
    for run in range(1,num_runs+1):
        res = pck.load(open(path + name + "_run" + str(run) + "/results.pkl", "rb"))
        corr_runs.append(res['corr_mapper'][1][0])
    corr_means.append(np.mean(np.array(corr_runs)))
    corr_vars.append(np.var(np.array(corr_runs)))
corrs_to_write = ['{0:.3g}'.format(corr_means[idx_c]) + ' $\\pm$ ' + '{0:.3g}'.format(corr_vars[idx_c]) for idx_c in range(len(corr_vars))]
f.write(' & '.join(corrs_to_write) + ' \\\\ \n')
f.write('\\hline \n')
for idx_s, score_name in enumerate(score_names):
    f.write(score_name + ' & ')
    scores_means, scores_vars = [], []
    for name in names:
        scores_runs = []
        for run in range(1,num_runs+1):
            res = pck.load(open(path + name + "_run" + str(run) + "/results.pkl", "rb"))
            scores = res['scores']
            scores_runs.append(scores[5][idx_s])
        scores_means.append(np.mean(np.array(scores_runs)))
        scores_vars.append(np.var(np.array(scores_runs)))
    scores_to_write = ['{0:.3g}'.format(scores_means[idx_sc]) + ' $\\pm$ ' + '{0:.3g}'.format(scores_vars[idx_sc]) for idx_sc in range(len(scores_means))]
    f.write(' & '.join(scores_to_write) + ' \\\\ \n')
f.write('\\hline \n')
f.write('\\end{tabular}')
f.close() 

