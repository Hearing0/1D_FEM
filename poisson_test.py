import numpy as np

def solve_poisson(num_elements, length=1.0):
    num_nodes = num_elements + 1
    h = length / num_elements
    stiffness = np.zeros((num_nodes, num_nodes))
    load = np.zeros(num_nodes)

    for element in range(num_elements):
        local_stiffness = np.array([[1, -1], [-1, 1]]) / h
        local_load = np.array([h / 2, h / 2])
        nodes = [element, element + 1]

        stiffness[np.ix_(nodes, nodes)] += local_stiffness
        load[nodes] += local_load

    # u(0) = u(length) = 0, so solve only for interior nodes.
    values = np.zeros(num_nodes)
    values[1:-1] = np.linalg.solve(
        stiffness[1:-1, 1:-1],
        load[1:-1],
    )

    x = np.linspace(0, length, num_nodes)
    exact = x * (length - x) / 2
    return x, values, exact