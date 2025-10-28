from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize.optimize import _status_message
from scipy._lib._util import check_random_state
from scipy.stats import entropy

#from Evolution_Simulator import Evolution_Simulator
from utils import pop2img

import random
import warnings
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
import matplotlib.pyplot as plt
seed=2
np.random.seed(seed)  # 随机种子

__all__ = ['differential_evolution']

_MACHEPS = np.finfo(np.float64).eps


def differential_evolution(img, func, bounds, args=(), strategy='best1bin',
                           maxiter=1000, popsize=15, tol=0.01,
                           mutation=(0.5, 1), recombination=0.7, seed=None,
                           callback=None, disp=False, polish=True,
                           init='latinhypercube', atol=0,early_stop=0,perturb_all=0, idx=None, img_rgb=None):    
    if args[0] == -1:
        print('Random:') # 未完成待续

    if args[0] == 0:
        print('DE:')
        solver = DifferentialEvolutionSolver(img, func, bounds, args=args, strategy=strategy, maxiter=maxiter,
                                             popsize=popsize, tol=tol, mutation=mutation, recombination=recombination,
                                             seed=seed, polish=polish, callback=callback, disp=disp, init=init,
                                             atol=atol)
        return solver.solve()
    elif args[0] == 1:
        print('DDE:')
        solver = DifferentialEvolutionSolver(img, func, bounds, args=args, strategy=strategy, maxiter=maxiter,
                                             popsize=popsize, tol=tol, mutation=mutation, recombination=recombination,
                                             seed=seed, polish=polish, callback=callback, disp=disp, init=init,
                                             atol=atol)
        return solver.dynamic_solve()
    elif args[0] == 2:
        print('ADE_pop2:')
        solver = DifferentialEvolutionSolver(img, func, bounds, args=args, strategy=strategy, maxiter=maxiter,
                                             popsize=popsize, tol=tol, mutation=mutation, recombination=recombination,
                                             seed=seed, polish=polish, callback=callback, disp=disp, init=init,
                                             atol=atol, early_stop=early_stop, perturb_all=perturb_all, idx=idx, img_rgb=img_rgb)
        return solver.AdaptiveDynamic_solve()
    


class DifferentialEvolutionSolver(object):
    """This class implements the differential evolution solver
    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'currenttobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
        The default is 'best1bin'
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * len(x)``
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        ``popsize * len(x)`` individuals (unless the initial population is
        supplied via the `init` keyword).
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        U[min, max). Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    seed : int or `np.random.RandomState`, optional
        If `seed` is not specified the `np.random.RandomState` singleton is
        used.
        If `seed` is an int, a new `np.random.RandomState` instance is used,
        seeded with `seed`.
        If `seed` is already a `np.random.RandomState` instance, then that
        `np.random.RandomState` instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Display status messages
    callback : callable, `callback(xk, convergence=val)`, optional
        A function to follow the progress of the minimization. ``xk`` is
        the current value of ``x0``. ``val`` represents the fractional
        value of the population convergence.  When ``val`` is greater than one
        the function halts. If callback returns `True`, then the minimization
        is halted (any polishing is still carried out).
    polish : bool, optional
        If True, then `scipy.optimize.minimize` with the `L-BFGS-B` method
        is used to polish the best population member at the end. This requires
        a few more function evaluations.
    maxfun : int, optional
        Set the maximum number of function evaluations. However, it probably
        makes more sense to set `maxiter` instead.
    init : str or array-like, optional
        Specify which type of population initialization is performed. Should be
        one of:
            - 'latinhypercube'
            - 'random'
            - array specifying the initial population. The array should have
              shape ``(M, len(x))``, where len(x) is the number of parameters.
              `init` is clipped to `bounds` before use.
        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space. 'random'
        initializes the population randomly - this has the drawback that
        clustering can occur, preventing the whole of parameter space being
        covered. Use of an array to specify a population could be used, for
        example, to create a tight bunch of initial guesses in an location
        where the solution is known to exist, thereby reducing time for
        convergence.
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    """

    # Dispatch of mutation strategy method (binomial or exponential).
    _binomial = {'best1bin': '_best1',
                 'randtobest1bin': '_randtobest1',
                 'currenttobest1bin': '_currenttobest1',
                 'best2bin': '_best2',
                 'rand2bin': '_rand2',
                 'rand1bin': '_rand1'}
    _exponential = {'best1exp': '_best1',
                    'rand1exp': '_rand1',
                    'randtobest1exp': '_randtobest1',
                    'currenttobest1exp': '_currenttobest1',
                    'best2exp': '_best2',
                    'rand2exp': '_rand2'}

    __init_error_msg = ("The population initialization method must be one of "
                        "'latinhypercube' or 'random', or an array of shape "
                        "(M, N) where N is the number of parameters and M>5")

    def __init__(self, img, func, bounds, args=(),
                 strategy='best1bin', maxiter=1000, popsize=15,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                 maxfun=np.inf, callback=None, disp=False, polish=True,
                 init='latinhypercube', atol=0, early_stop=0, perturb_all=0, idx=None, img_rgb=None):
        self.perturb_all = perturb_all  # 目标攻击成功的种群数
        self.idx = idx
        self.img_rgb = img_rgb
        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy
        self.callback = callback
        self.polish = polish

        # relative and absolute tolerances for convergence
        self.tol, self.atol = tol, atol

        # Mutation constant should be in [0, 2). If specified as a sequence
        # then dithering is performed.
        self.scale = mutation
        if (not np.all(np.isfinite(mutation)) or
                np.any(np.array(mutation) >= 2) or
                np.any(np.array(mutation) < 0)):
            raise ValueError('The mutation constant must be a float in '
                             'U[0, 2), or specified as a tuple(min, max)'
                             ' where min < max and min, max are in U[0, 2).')

        self.dither = None
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        self.cross_over_probability = recombination
        self.img = img
        self.func = func
        self.args = args

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        self.limits = np.array(bounds, dtype='float').T
        if (np.size(self.limits, 0) != 2 or not
        np.all(np.isfinite(self.limits))):
            raise ValueError('bounds should be a sequence containing '
                             'real valued (min, max) pairs for each value'
                             ' in x')

        if maxiter is None:  # the default used to be None
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:  # the default used to be None
            maxfun = np.inf
        self.maxfun = maxfun

        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.parameter_count = np.size(self.limits, 1)
        self.random_number_generator = check_random_state(seed)

        # popsize为单个种群大小
        self.num_population_members = max(5, popsize)

        self.population_shape = (self.num_population_members,
                                 self.parameter_count)
        N,M,p = init.shape
        assert (M,p) == self.population_shape, 'Pop shape wrong!'
        self.popnum = N
        self.population_energies = np.ones((N, M))

        self._nfev = 0
        
        self.init_population_array(init)

        self.disp = disp
        self.last_pop = None
        self.stop_evolution_community = [False for i in range(self.popnum)]
        self.cross_mutate_times = [0 for i in range(self.popnum)]
        self.early_stop = early_stop
        self.success_pop = [False for i in range(self.popnum)]
        self.success_num = 0

    def init_population_array(self, init):
        """
        Initialises the population with a user specified population.
        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial population. The array should
            have shape (N, M, len(x)), where len(x) is the number of parameters.
            N is the population number, M is the capacity of each population.
            By default, each population should have the same capacity.
            The population is clipped to the lower and upper `bounds`.
        """
        # make sure you're using a float array
        N,M,p = init.shape
        popN = np.asfarray(init)
        self.population = np.zeros((N,M,p), dtype=np.float32)
        self.population_mask = np.ones((N,M * p // 3), dtype=np.float32)
        self.merge_population = np.zeros(N, dtype=np.float32)
        for i,popn in enumerate(popN):

            if (np.size(popn, 0) < 5 or
                    popn.shape[1] != self.parameter_count or
                    len(popn.shape) != 2):
                raise ValueError("The population supplied needs to have shape"
                                " (M, len(x)), where M > 4.")

            # scale values and clip to bounds, assigning to population
            self.population[i] = np.clip(self._unscale_parameters(popn), 0, 1)

            # reset population energies
            self.population_energies[i] = (np.ones(self.num_population_members) *
                                        np.inf)
        self.num_population_members = np.size(self.population[0], 0)
        self.population_shape = (self.num_population_members,
                                    self.parameter_count)
        # reset number of function evaluations counter
        self._nfev = 0

    @property
    def x(self):
        """
        The best solution from the solver
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        """
        min_x = self.population[0][0]
        min_v = self.population_energies[0][0]
        for i in range(self.popnum):
            if min_v > self.population_energies[i][0]:
                min_x = self.population[i][0]
                min_v = self.population_energies[i][0]
        return self._scale_parameters(min_x)
    
    def xs(self, L=None):
        # 返回最小的L个种群最小值
        if L is None:
            L = self.popnum
        xs = []
        energies = []
        result = []
        for i in range(self.popnum):
            energies.append(self.population_energies[i][0])
            xs.append(self._scale_parameters(self.population[i][0]))
        for i in range(L):
            index = energies.index(min(energies))
            result.append(xs[index])
            del xs[index]
            del energies[index]
        result = np.concatenate(result, axis=0)
        return result

    @property
    def convergence(self):
        """
        The standard deviation of the population energies divided by their
        mean.
        """
        return (np.std(self.population_energies) /
                np.abs(np.mean(self.population_energies) + _MACHEPS))

    def Fast(self, begin=2):
        #3
        # do the optimisation.
        status_message = 'Init status'
        for nit in range(begin, begin + 1):
            # evolve the population by a generation
            try:
                next(self)
            except StopIteration:
                status_message = _status_message['maxfev']
                break
            
            if self.disp:
                print(f"differential_evolution step {nit}:")
                for i in range(self.popnum):
                    print(f"Pop {i}: min = {self.population_energies[i][0]:.2f}, max = {max(self.population_energies[i]):.2f}   {'---stopped' if self.stop_evolution_community[i] else ''}")
        else:
            status_message = _status_message['maxiter']

        xs = self.xs()
        self.success = self.callback(xs, convergence=0)
        DE_result = OptimizeResult(
            x=xs,
            fun=self.population_energies[0],
            nfev=self._nfev,
            nit=nit + 1,
            message=status_message,
            success=self.success)

        return DE_result

    def AdaptiveDynamic_solve(self):
        nit, self.success = 0, False
        status_message = _status_message['success']
        # 先让各个种群内部进化
        # 第一步，计算初始能量
        for i in range(self.popnum):
            if self.args[2]<-100:
                print('Fast mode:', 0)
                self._calculate_population_energies()
            else:
                if np.all(np.isinf(self.population_energies[i])):
                    self.itersize = max(0, min(len(self.population[i]), self.maxfun - self._nfev + 1))
                    candidates = self.population[i][:self.itersize]
                # 第0代
                minval, lowest_energy = 999999999, 999999999
                for candidate in range(self.itersize):
                    parameter = np.array(self._scale_parameters(candidates[candidate]))  # TODO: vectorize
                    energy = self.func(parameter, 1)
                    self.population_energies[i][candidate] = energy
                    self._nfev += 1
                    if energy < lowest_energy:
                        lowest_energy = energy
                        minval = candidate
                        if self.disp:
                            print("differential_evolution step %d-%d: f(x)= %g"
                                % (nit, candidate, lowest_energy))
                        # should the solver terminate?
                        convergence = self.convergence
                        self.success = self.callback(self._scale_parameters(self.population[i][minval]),
                                                    convergence=self.tol / convergence) is True
                        if (self.perturb_all == 0) and self.success:
                            status_message = ('callback function requested stop early by returning True')
                            DE_result = OptimizeResult(
                                x=self._scale_parameters(self.population[i][minval]),
                                fun=self.population_energies[i][minval],
                                nfev=self._nfev,
                                nit=nit + (candidate + 1) / self.itersize,
                                message=status_message,
                                success=self.success)
                            return DE_result
                        elif self.perturb_all > 0 and self.success:
                            self.success_pop[i] = True
                            self.success_num += 1
                            break
                            
                self.population_energies[i][minval] = self.population_energies[i][0]
                self.population_energies[i][0] = lowest_energy
                self.population[i][[0, minval], :] = self.population[i][[minval, 0], :]
        # 第0代初始化结束
        
        for iteration in range(1, self.maxiter + 1):                              
            print(f'Fast mode: {iteration}')
            result = self.Fast(begin=iteration)
            if all(self.stop_evolution_community):
                return result
            # Display the pop distribution
            parameters = []
            for i in range(self.popnum):
                parameters.append(np.array([self._scale_parameters(trial) for trial in self.population[i]]))
            parameters = np.concatenate(parameters, axis=0)
            imgH = self.limits[1][0] + 1
            imgW = self.limits[1][1] + 1
            pop2img(parameters, int(imgH), int(imgW),iteration, self.idx, image=self.img_rgb)    

            # # 强迫到最大迭代次数再停止，看看能否涨ED
            # if result!= None and result.success == True:
            #     return result
        else:   # 达到最大迭代次数，直接返回最优结果
                result.success = True
        return result
    
    def _calculate_population_energies(self):
        """
        Calculate the energies of all the population members at the same time.
        Puts the best member in first place. Useful if the population has just
        been initialised.
        """

        ##############
        ## CHANGES: self.func operates on the entire parameters array
        ##############
        self.itersize = max(0, min(len(self.population), self.maxfun - self._nfev + 1))
        candidates = self.population[:self.itersize]
        parameters = np.array([self._scale_parameters(c) for c in candidates])  # TODO: vectorize
        energies = self.func(parameters, self.num_population_members)
        self.population_energies = energies
        self._nfev += self.itersize

        minval = np.argmin(self.population_energies)

        # put the lowest energy into the best solution position.
        lowest_energy = self.population_energies[minval]
        self.population_energies[minval] = self.population_energies[0]
        self.population_energies[0] = lowest_energy

        self.population[[0, minval], :] = self.population[[minval, 0], :]

    def __iter__(self):
        return self

    def __next__(self):
        """
        Evolve the population by a single generation
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # the population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies
        # if np.all(np.isinf(self.population_energies)):
        #     self._calculate_population_energies()

        if self.dither is not None:
            self.scale = (self.random_number_generator.rand()
                          * (self.dither[1] - self.dither[0]) + self.dither[0])

        self.itersize = self.num_population_members
        trials = np.array([self._mutate(c) for c in range(self.itersize)])
        trials = np.transpose(trials, (1,0,2))
        KLs = [0 for i in range(self.popnum)]
        for i,trial in enumerate(trials):   # 遍历N个种群
            if self.stop_evolution_community[i]:    # 此种群无法进化，直接跳过
                continue
            stride = 1 / self.popnum
            w_min = i * stride
            w_max = (i+1) * stride
            w_limit = (w_min, w_max)
            for j, t in enumerate(trial): 
                self._ensure_constraint(t, w_limit=w_limit)  # 限制种群的范围，避免越界
                if self.args[1] != -1:
                    self.DEC(self.img, t, self.args)    # 迫使像素值为0或255

            # # 如果种群能量太高，则重新用单点计算能量
            # if self.population_energies[0][0] == 10:
            #     parameters = np.array([self._scale_parameters(t) for t in trial])
            # else:   # 否则用多点计算能量
            parameters = []
            best_pixels = [self._scale_parameters(self.population[j][0]) for j in range(self.popnum)]
            for t in trial:
                parameter = self._scale_parameters(t)
                for j in range(self.popnum):
                    if j != i :
                        parameter = np.hstack((parameter, best_pixels[j])) 
                parameters.append(parameter)
            parameters = np.array(parameters)
            energies = self.func(parameters, self.itersize)

            # if one community has no change, stop it
            kl_divergence = entropy(self.population_energies[i]+0.01, energies+0.01)
            KLs[i] = kl_divergence
            if kl_divergence == 0. or min(energies) >= self.population_energies[i, 0]:
                # early stop
                self.stop_evolution_community[i] = True if self.early_stop == 1 else False
                continue

            for candidate, (energy, t) in enumerate(zip(energies, trial)):
                # if the energy of the trial candidate is lower than the
                # original population member then replace it
                if energy < self.population_energies[i][candidate]:
                    self.population[i][candidate] = t
                    self.population_energies[i][candidate] = energy

                    # if the trial candidate also has a lower energy than the
                    # best solution, then replace that as well
                    if energy < self.population_energies[i][0]:
                        self.population_energies[i][0] = energy
                        self.population[i][0] = t
            self._nfev += self.num_population_members
        return 

    def next(self):
        """
        Evolve the population by a single generation
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # next() is required for compatibility with Python2.7.
        return self.__next__()

    def _scale_parameters(self, trial):
        """
        scale from a number between 0 and 1 to parameters.
        """
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        """
        scale from parameters to a number between 0 and 1.
        """
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def DEC(self, img, vec, args):
        # cycle through each variable in vector
        if args[1] == 1:
            for i in range(2, len(vec), 3):
                if vec[i] > 0.5:
                    vec[i] = 1
                else:
                    vec[i] = 0
        elif args[1] == 2:
            vec_scale = self._scale_parameters(vec)
            for i in range(2, len(vec), 3):
                if img[0, 0, int(vec_scale[i - 2]), int(vec_scale[i - 1])] > 0:
                    vec[i] = 0
                else:
                    vec[i] = 1
        elif args[1] == 0:
            for i in range(2, len(vec), 3):
                vec[i] = 0
        elif args[1] == 255:
            for i in range(2, len(vec), 3):
                vec[i] = 1

    def _ensure_constraint(self, trial, w_limit=None, h_limit=None):
        """
        make sure the parameters lie between the limits
        限制W和H在min和max之间
        """
        if w_limit is not None:
            wmin, wmax = w_limit
            div = wmax - wmin
            for index in np.where((trial < wmin) | (trial > wmax))[0]:
                if index%3==1: trial[index] = wmin + self.random_number_generator.rand() % div
        if h_limit is not None:
            hmin, hmax = h_limit
            div = hmax - hmin
            for index in np.where((trial < hmin) | (trial > hmax))[0]:
                if index%3==0: trial[index] = hmin + self.random_number_generator.rand() % div
        for index in np.where((trial < 0) | (trial > 1))[0]:
            trial[index] = self.random_number_generator.rand()
    

    def _mutate(self, candidate):
        """
        create a trial vector based on a mutation strategy
        返回每个种群的trial=(popnum, pixel_num*3).
        """
        trials = np.copy(self.population[:, candidate])

        rng = self.random_number_generator

        fill_point = rng.randint(0, self.parameter_count)

        if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprimes = self.mutation_func(candidate, self._select_samples(candidate, 5))
        else:
            bprimes = self.mutation_func(self._select_samples(candidate, 5))

        if self.strategy in self._binomial:
            for i,bprime in enumerate(bprimes):
                crossovers = rng.rand(self.parameter_count)
                crossovers = crossovers < self.cross_over_probability
                # the last one is always from the bprime vector for binomial
                # If you fill in modulo with a loop you have to set the last one to
                # true. If you don't use a loop then you can have any random entry
                # be True.
                crossovers[fill_point] = True
                trials[i] = np.where(crossovers, bprime, trials[i])
            return trials

        elif self.strategy in self._exponential:
            print(f'strategy {self.strategy} not implement yet!')
            return bprime
            # i = 0
            # while (i < self.parameter_count and
            #        rng.rand() < self.cross_over_probability):
            #     trial[fill_point] = bprime[fill_point]
            #     fill_point = (fill_point + 1) % self.parameter_count
            #     i += 1

            # return trial

    def _best1(self, samples):
        """
        best1bin, best1exp
        返回各个种群的最优个体变异结果
        """
        r0, r1 = samples[:2]
        ret_values = []
        for i in range(self.popnum):
            # 交叉变异，已废弃
            # if self.stop_evolution_community[i] and self.success_pop[i] == False:
            #     another_pop = random.randint(0, self.popnum-1)
            #     ret = self.population[i][0] + self.scale * (self.population[another_pop][r0] - self.population[another_pop][r1])
            # elif self.success_pop[i] == False:
            if self.success_pop[i] == False:
                ret = self.population[i][0] + self.scale * (self.population[i][r0] - self.population[i][r1])
            else:   # 攻击成功后是否还要继续迭代？
                ret = self.population[i][0] + self.scale * (self.population[i][r0] - self.population[i][r1])
            ret_values.append(ret)
        return ret_values

    def _rand1(self, samples):
        """
        rand1bin, rand1exp
        """
        r0, r1, r2 = samples[:3]
        return (self.population[r0] + self.scale *
                (self.population[r1] - self.population[r2]))

    def _randtobest1(self, samples):
        """
        randtobest1bin, randtobest1exp
        """
        r0, r1, r2 = samples[:3]
        bprime = np.copy(self.population[r0])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r1] -
                                self.population[r2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        """
        currenttobest1bin, currenttobest1exp
        """
        r0, r1 = samples[:2]
        bprime = (self.population[candidate] + self.scale *
                  (self.population[0] - self.population[candidate] +
                   self.population[r0] - self.population[r1]))
        return bprime

    def _best2(self, samples):
        """
        best2bin, best2exp
        """
        r0, r1, r2, r3 = samples[:4]
        bprime = (self.population[0] + self.scale *
                  (self.population[r0] + self.population[r1] -
                   self.population[r2] - self.population[r3]))

        return bprime

    def _rand2(self, samples):
        """
        rand2bin, rand2exp
        """
        r0, r1, r2, r3, r4 = samples
        bprime = (self.population[r0] + self.scale *
                  (self.population[r1] + self.population[r2] -
                   self.population[r3] - self.population[r4]))

        return bprime

    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(self.num_population_members),
        without replacement.  You can't have the original candidate either.
        """
        idxs = list(range(self.num_population_members))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs
