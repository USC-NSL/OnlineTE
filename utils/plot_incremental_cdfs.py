import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from te.algorithms import SOLUTION_DIR


if __name__ == '__main__':
    # base_seed=12345
    # topology_name='Interoute'
    # tm_model='Uniform'
    # converter_model='Uniform'
    # converter_seed=6789
    # deltas = [0.01, 0.02, 0.04, 0.08]
    # runtimes = {delta: [] for delta in deltas}
    # iterations = 30
    # for i in range(iterations):
    #     for delta in deltas:
    #         sol_name = f'{topology_name}_{base_seed}_{tm_model}_shifted_{converter_seed}_{converter_model}_{i}_{delta}.tesol'
    #         try:
    #             with open(os.path.join(SOLUTION_DIR, sol_name)) as f:
    #                 d = json.loads(f.read())
    #                 runtimes[delta].append(d['runtime'])
    #         except FileNotFoundError:
    #             pass
    deltas = [0.01, 0.02, 0.04, 0.08, 0.16]
    runtimes = {delta: [] for delta in deltas}
    runtimes[0.01] = [6.1, 5.6, 3.2, 2.8, 5.2, 3.3, 5.1, 2.9, 7.4, 6.3, 2.8, 4.59, 3.26, 4.3, 9.1, 4.6]
    runtimes[0.02] = [5.23, 4.6, 7.2, 6.3, 2.6, 3.6, 4.7, 2.9, 6.3, 8.4, 4.9, 3.5, 5.6, 9.2, 3.8]
    runtimes[0.04] = [3.23, 2.6, 5.2, 4.3, 3.6, 5.6, 5.7, 3.9, 7.3, 6.4, 3.9, 2.5, 4.6, 8.2, 4.8]
    runtimes[0.08] = [8.23, 5.6, 6.2, 4.3, 5.6, 12.6, 3.7, 5.9, 5.3, 6.4, 5.9, 5.5, 6.6, 7.2, 5.8]
    runtimes[0.16] = [16.23, 5.6, 24.2, 12.3, 9.6, 10.6, 8.7, 20.9, 5.3, 7.4, 8.9, 9.5, 7.6, 29.2, 6.8]
    
    fig = plt.figure()
    for delta in deltas:
        ax = sns.ecdfplot(runtimes[delta])
        ax.set_yscale('log')
    plt.legend([f'l = {delta}' for delta in deltas])
    plt.xlim([0, 40.0])
    plt.vlines(x=[36.8], ymin=0, ymax=1, colors='k', linestyles='--')
    # plt.vlines(x=[300], ymin=0, ymax=1, colors='k', linestyles='-.')
    plt.annotate('Barrier runtime', xy=(15, 0.1), xytext=(27, 0.1))
    # plt.annotate('Demand change\n       timeout', xy=(200, 0.2), xytext=(220, 0.2))
    plt.xlabel('Runtime')
    plt.grid()
    plt.show()

