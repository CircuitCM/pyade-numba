import numpy as np

import gopt.de
import gopt.jade
import gopt.sade
import gopt.shade_ext
import gopt.lshade
import gopt.ilshade
import gopt.jso
import gopt.lshadecnepsin
import gopt.mpede
import gopt.commons

import random

random.seed(0)
np.random.seed(0)


def test_init_population():
    for i in range(10):
        pop_size = random.randint(3, 1000)
        ind_size = random.randint(3, 100)
        a = random.randint(-1000, 0)
        b = random.randint(0, 1000)
        pop = gopt.commons.init_population(pop_size, ind_size, [[a, b] * ind_size])
        assert pop.shape == (pop_size, ind_size)


def test_boundaries():
    for i in range(10):
        pop_size = random.randint(3, 1000)
        ind_size = random.randint(3, 100)
        a = random.randint(-1000, 0)
        b = random.randint(0, 1000)
        pop = gopt.commons.init_population(pop_size, ind_size, [[a, b] * ind_size])

        a /= 2
        b /= 2
        pop = gopt.commons.keep_bounds(pop, np.array([[a, b] * ind_size]))
        assert pop.min() >= a
        assert pop.min() <= b


def my_fitness(x: np.ndarray):
    return (x**2).sum()

def my_callback(**kwargs):
    global output
    if not output['initialized']:
        output['min']: np.min(kwargs['fitness'])
        output['mean']: np.mean(kwargs['fitness'])
        output['max']: np.max(kwargs['fitness'])
        ordered = sorted(kwargs['fitness'])
        output['median'] = ordered[len(ordered) // 2]

    if np.random.rand() < 0.3:
        new_min = np.min(kwargs['fitness'])
        new_mean = np.mean(kwargs['fitness'])
        new_max = np.max(kwargs['fitness'])
        ordered = sorted(kwargs['fitness'])
        new_median = ordered[len(ordered) // 2]

        assert (new_min <= output['min'])
        assert (new_mean <= output['mean'])
        assert (new_max <= output['max'])
        assert (new_median <= output['median'])

        output['min'] = new_min
        output['mean'] = new_mean
        output['max'] = new_max
        output['median'] = new_median


def test_algorithms():
    global output

    algorithms = [gopt.de, gopt.sade, gopt.jade, gopt.shade_ext, gopt.lshade, gopt.lshadecnepsin, gopt.ilshade,
                  gopt.jso, gopt.mpede]

    for algorithm in algorithms:
        params = algorithm.get_default_params(dim=10)
        params['max_evals'] = 4000
        params['bounds'] = np.array([[-100, 100]] * params['individual_size'])
        params['func'] = my_fitness
        params['callback'] = my_callback

        output = {'initialized': False, 'min': np.inf, 'max': np.inf, 'mean': np.inf, 'median': np.inf}

        algorithm.apply(**params)
