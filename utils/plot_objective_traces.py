import os
import matplotlib.pyplot as plt
from typing import List


this_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(this_dir, '..')


def get_objective_value_trace(path: str) -> List[float]:
    with open(path) as trace:
        for line_str in trace:
            line_str = line_str.strip()
            if line_str.startswith('objective_value:'):
                values = line_str.split()[-1]
                return [float(item) for item in values.split(',')]


if __name__ == '__main__':
    k1 = get_objective_value_trace(os.path.join(root_dir, 'res_k=1.txt'))
    k2 = get_objective_value_trace(os.path.join(root_dir, 'res_k=2.txt'))
    k4 = get_objective_value_trace(os.path.join(root_dir, 'res_k=4.txt'))
    k8 = get_objective_value_trace(os.path.join(root_dir, 'res_k=8.txt'))
    k16 = get_objective_value_trace(os.path.join(root_dir, 'res_k=16.txt'))

    plt.plot(k1)
    plt.plot(k2)
    plt.plot(k4)
    plt.plot(k8)
    plt.plot(k16)
    plt.legend(['k=1', 'k=2', 'k=4', 'k=8', 'k=16'])
    plt.xlabel('Maximum Link Utilization')
    plt.ylabel('Epoch')
    plt.show()
