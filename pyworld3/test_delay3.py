# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:28:23 2022

@author: Tim Schell
"""

import numpy as np

from .specials import Delay3
from .utils import requires

class test_delay3:

    def __init__(self, year_min=1900, year_max=2100, dt=1,
                 verbose=False):

        self.dt = dt
        self.year_min = year_min
        self.year_max = year_max
        self.verbose = verbose
        self.length = self.year_max - self.year_min
        self.n = int(self.length / self.dt)
        self.time = np.arange(self.year_min, self.year_max, self.dt)

    def init_resource_constants(self, delay_timex = 20):
        """
        Initialize the constant parameters of the resource sector. Constants
        and their unit are documented above at the class level.

        """

        self.delay_timex = delay_timex

    def init_resource_variables(self):
        """
        Initialize the state and rate variables of the resource sector
        (memory allocation). Variables and their unit are documented above at
        the class level.

        """
        self.array = np.full((self.n,), np.nan)
        self.array_geglättet = np.full((self.n,), np.nan)
        self.delay_time = np.full((self.n,), 20)

    def set_resource_delay_functions(self, method="euler"):
        """
        Set the linear smoothing and delay functions of the 1st or the 3rd
        order, for the resource sector. One should call
        `self.set_resource_delay_functions` after calling
        `self.init_resource_constants`.

        Parameters
        ----------
        method : str, optional
            Numerical integration method: "euler" or "odeint". The default is
            "euler".
        """
        
        #neu hinzugefügt:
        var_delay3 = ["array_geglättet","array", "delay_time"]
        for var_ in var_delay3:
            func_delay = Delay3(getattr(self, var_.lower()),
                                self.dt, self.time, method=method)
            setattr(self, "delay3_"+var_.lower(), func_delay)

        """
        Delay Funktion aus Pollution-class kopiert und Variablen angepasst.
        """

    def loop0_resource(self, alone=False):
        """
        Run a sequence to initialize the resource sector (loop with k=0).

        Parameters
        ----------
        alone : boolean, optional
            if True, run the sector alone with exogenous inputs. The default
            is False.

        """
        
        self._update_delay_test(0)

    def loopk_resource(self, j, k, jk, kl, alone=False):
        """
        Run a sequence to update one loop of the resource sector.

        Parameters
        ----------
        alone : boolean, optional
            if True, run the sector alone with exogenous inputs. The default
            is False.

        """

        self._update_delay_test(k)

    def run_delay_test(self):
        """
        Run a sequence of updates to simulate the resource sector alone with
        exogenous inputs.

        """
        self.redo_loop = True
        while self.redo_loop:
            self.redo_loop = False
            self.loop0_resource(alone=True)
        for k_ in range(1, self.n):
            self.redo_loop = True
            while self.redo_loop:
                self.redo_loop = False
                if self.verbose:
                    print("go loop", k_)
                self.loopk_resource(k_-1, k_, k_-1, k_, alone=True)

    @requires(["array"])
    def _update_delay_test(self, k):
        """
        State variable, requires previous step only
        """
        if k == 0:
            self.array[k] = 1
        if k > 0:
            self.array[k] = self.array[k-1] + 1
        
        self.array_geglättet[k] = self.delay3_array(k, self.delay_time[k])
    