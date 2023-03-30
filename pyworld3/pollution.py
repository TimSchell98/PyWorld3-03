# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:35:01 2022

PyWorld3 2004 update
Completety renewed with the original STELLA version of World3

@author: Tim Schell
"""

# -*- coding: utf-8 -*-

# © Copyright Charles Vanwynsberghe (2021)

# Pyworld3 is a computer program whose purpose is to run configurable
# simulations of the World3 model as described in the book "Dynamics
# of Growth in a Finite World".

# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".

# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.

# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import os
import json

from scipy.interpolate import interp1d
import numpy as np

from .specials import Dlinf3, clip, switch, Delay3
from .utils import requires


class Pollution:
    """
    Persistent Pollution sector. Can be run independantly from other sectors
    with exogenous inputs. The initial code is defined p.478.
    
    Completely rebuild for the 2004 update.

    Examples
    --------
    Running the persistent pollution sector alone requires artificial
    (exogenous) inputs which should be provided by the other sectors. Start
    from the following example:

    >>> pol = Pollution()
    >>> pol.set_pollution_table_functions()
    >>> pol.init_pollution_variables()
    >>> pol.init_pollution_constants()
    >>> pol.set_pollution_delay_functions()
    >>> pol.init_exogenous_inputs()
    >>> pol.run_pollution()

    Parameters
    ----------
    year_min : float, optional
        start year of the simulation [year]. The default is 1900.
    year_max : float, optional
        end year of the simulation [year]. The default is 2100.
    dt : float, optional
        time step of the simulation [year]. The default is 1.
    pyear : float, optional
        implementation date of new pol
        icies [year]. The default is 1975.
    pyear_pp_tech : float, optional
        implementation date of new pollution policies [year]. The default is 4000.
    verbose : bool, optional
        print information for debugging. The default is False.

    Attributes
    ----------
    apct : float [year]
        Air pollution change time. Default 4000
    io70 : float
        Industrial Output in 1970. Default is 7.9e11
    imef : float
        Industrial material emission factor. Default is 0.1
    imti : float
        Industrial material toxic index. Default is 10
    frpm : float
        Fraction res pers mtl. Default is 0.02
    ghup : float
        Gha per unit of pollution. Default is 4e-9
    faipm : float
        Fraction agricultural input pers mtl. Default is 0.001
    amti : float
        Agricultural material toxicity index. Default is 1.
    pptd : float [years]
        Persistent pollution transmission delay. Default is 20
    ahl70 : float
        Assimilation half life 1970. Default is 1.5
    pp70: float
        Persistent pollution in 1970. Default is 1.36e8
    dppolx : float
        Desired persistent pollution index. Default is 1.2
    tdt : float [years]
        Technology development time. Default is 20 
    apfay : numpy.ndarray
        Air pollution factor on agricultural yield.
    ymap1 : numpy.ndarray
        Yield multiplier air poll 1
    ymap2 : numpy.ndarray
        Yield multiplier air poll 2
    fio70 : numpy.ndarray
        Fraction Industrial Output over 1970
    io : numpy.ndarray
        Industrial Output. From Capital Sector
    pii : numpy.ndarray
        Pollution intensity index
    pop : numpy.ndarray
        Population. From population sector
    pcrum : numpy.ndarray
        Per capita resource use multiplier
    ppgi : numpy.ndarray
        persistent pollution generation industry
    aiph : numpy.ndarray
        Agricultural inputs per hectare
    ppga : numpy.ndarray
        Persistent Pollution Gen Agricultural
    ppgr : numpy.ndarray
        Persistent pollution generation rate
    ppar : numpy.ndarray
        Persistent Pollution appear rate.
    ppasr : numpy.ndarray
        Persistent Pollution assimilation rate
    pp : numpy.ndarray
        Persistent pollution
    ppasr : numpy.ndarray
        Persistant Pollution assimilation rate
    ahl : numpy.ndarray
        Assimilation half life
    ahlm : numpy.ndarray
        Assimilation half life multiplier
    ppolx : numpy.ndarray
        Persistent Pollution Index
    pptc : numpy.ndarray
        Persistent Pollution Tech Change
    pptcm : numpy.ndarray
        Persistent pollution tech change multiplier
    pptcr : numpy.ndarray
        Persistent polllution tech change rate
    ppt : numpy.ndarray
        Persistent pollution tech
    ppgf1 : numpy.ndarray
        Persistent Pollution Gen Fact 1
    ppgf2 : numpy.ndarray
        Persistent Pollution Gen Fact 2
    ppgf : numpy.ndarray
        Persistent pollution generation factor
    pptmi : numpy.ndarray
        Persistent pollution tech multiplier icor COPM
    abl : numpy.ndarray
        Absorbtion Land in Gha
    ef : numpy.ndarray
        Human ecological footprint
    """

    def __init__(self, year_min=1900, year_max=2100, dt=1, pyear=1975, pyear_pp_tech = 4000,
                 verbose=False):
        
        self.pyear = pyear
        self.pyear_pp_tech = pyear_pp_tech
        self.dt = dt
        self.year_min = year_min
        self.year_max = year_max
        self.verbose = verbose
        self.length = self.year_max - self.year_min
        self.n = int(self.length / self.dt)
        self.time = np.arange(self.year_min, self.year_max, self.dt)

    def init_pollution_constants(self,pp19 = 2.5e7, apct = 4000.0, io70 = 7.9e11 ,imef = 0.1 ,imti = 10.0 ,frpm = 0.02
                                 ,ghup = 4e-9 ,faipm = 0.001 ,amti = 1.0 ,pptd = 20.0
                                 ,ahl70 = 1.5 ,pp70 = 1.36e8, dppolx = 1.2 ,tdt = 20.0, ppgf1 = 1.0):
        """
        Initialize the constant parameters of the pollution sector. Constants
        and their unit are documented above at the class level.

        """
        self.pp19 = pp19
        self.apct = apct
        self.io70 = io70
        self.imef = imef
        self.imti = imti
        self.frpm = frpm
        self.ghup = ghup
        self.faipm = faipm
        self.amti = amti
        self.pptd = pptd
        self.ahl70 = ahl70
        self.pp70 = pp70
        self.dppolx = dppolx
        self.tdt= tdt
        self.ppgf1 = ppgf1
 
    def init_pollution_variables(self):
        """
        Initialize the state and rate variables of the pollution sector
        (memory allocation). Variables and their unit are documented above at
        the class level.

        """
        self.apfay = np.full((self.n,), np.nan)
        self.ymap1 = np.full((self.n,), np.nan)
        self.ymap2 = np.full((self.n,), np.nan)
        self.fio70 = np.full((self.n,), np.nan)
        self.pii = np.full((self.n,), np.nan)
        self.pcrum = np.full((self.n,), np.nan)
        self.ppgi = np.full((self.n,), np.nan)
        self.aiph = np.full((self.n,), np.nan)
        self.ppga = np.full((self.n,), np.nan)
        self.ppgr = np.full((self.n,), np.nan)
        self.ppar = np.full((self.n,), np.nan)
        self.ppasr = np.full((self.n,), np.nan)
        self.pp = np.full((self.n,), np.nan)
        self.ppasr = np.full((self.n,), np.nan)
        self.ahl = np.full((self.n,), np.nan)
        self.ahlm = np.full((self.n,), np.nan)
        self.ppolx = np.full((self.n,), np.nan)
        self.pptc = np.full((self.n,), np.nan)
        self.pptcm = np.full((self.n,), np.nan)
        self.pptcr = np.full((self.n,), np.nan)
        self.ppt = np.full((self.n,), np.nan)
        self.ppgf2 = np.full((self.n,), np.nan)
        self.ppgf = np.full((self.n,), np.nan)
        self.pptmi = np.full((self.n,), np.nan)
        self.abl = np.full((self.n,), np.nan)
        self.ef = np.full((self.n,), np.nan)

    def set_pollution_delay_functions(self, method="euler"):
        """
        Set the linear smoothing and delay functions of the 1st or the 3rd
        order, for the pollution sector. One should call
        `self.set_pollution_delay_functions` after calling
        `self.init_pollution_constants`.

        Parameters
        ----------
        method : str, optional
            Numerical integration method: "euler" or "odeint". The default is
            "euler".

        """
            
        var_dlinf3 = ["PPGR", "PPT"]
        for var_ in var_dlinf3:
            func_delay = Dlinf3(getattr(self, var_.lower()),
                                self.dt, self.time, method=method)
            setattr(self, "dlinf3_"+var_.lower(), func_delay)

    def set_pollution_table_functions(self, json_file=None):
        """
        Set the nonlinear functions of the pollution sector, based on a json
        file. By default, the `functions_table_world3.json` file from pyworld3
        is used.

        Parameters
        ----------
        json_file : file, optional
            json file containing all tables. The default is None.

        """
        if json_file is None:
            json_file = "./functions_table_world3.json"
            json_file = os.path.join(os.path.dirname(__file__), json_file)
        with open(json_file) as fjson:
            tables = json.load(fjson)

        func_names = ["PCRUM","AHLM","PPTCM","PPTMI","YMAP1","YMAP2"]
        for func_name in func_names:
            for table in tables:
                if table["y.name"] == func_name:
                    func = interp1d(table["x.values"], table["y.values"],
                                    bounds_error=False,
                                    fill_value=(table["y.values"][0],
                                                table["y.values"][-1]))
                    setattr(self, func_name.lower()+"_f", func)

    def init_exogenous_inputs(self):
        """
        Initialize all the necessary constants and variables to run the
        pollution sector alone. These exogenous parameters are outputs from
        the 4 other remaining sectors in a full simulation of World3.

        """

        # constants
        self.swat = 0
        self.tdd = 10
        self.pd = 5
        self.cio = 100
        self.lt = self.year_min + 500
        self.lt2 = self.year_min + 500
        # variables
        self.pcrum = np.full((self.n,), np.nan)
        self.pop = np.full((self.n,), np.nan)
        self.aiph = np.full((self.n,), np.nan)
        self.al = np.full((self.n,), np.nan)
        self.pcti = np.full((self.n,), np.nan)
        self.pctir = np.full((self.n,), np.nan)
        self.pctcm = np.full((self.n,), np.nan)
        self.plmp = np.full((self.n,), np.nan)
        self.lmp = np.full((self.n,), np.nan)
        self.lmp1 = np.full((self.n,), np.nan)
        self.lmp2 = np.full((self.n,), np.nan)
        self.lfdr = np.full((self.n,), np.nan)
        self.lfdr1 = np.full((self.n,), np.nan)
        self.lfdr2 = np.full((self.n,), np.nan)
        self.ppgf22 = np.full((self.n,), np.nan)
        #2004 version added:
        self.io = np.full((self.n,), np.nan)
        self.io1 = np.full((self.n,), np.nan)
        self.io11 = np.full((self.n,), np.nan)
        self.io12 = np.full((self.n,), np.nan)
        self.io2 = np.full((self.n,), np.nan)
        self.iopc = np.full((self.n,), np.nan)
        self.uil = np.full((self.n,), 8.2e6)
        
        # tables
        func_names = ["PCRUM", "POP", "AIPH", "AL", "PCTCM", "LMP1", "LMP2",
                      "LFDR1", "LFDR2"]
        y_values = [[_ * 10**-2 for _ in [17, 30, 52, 78, 138, 280, 480, 660,
                                          700, 700, 700]],
                    [_ * 10**8 for _ in [16, 19, 22, 31, 42, 53, 67, 86, 109,
                                         139, 176]],
                    [6.6, 11, 20, 34, 57, 97, 168, 290, 495, 845, 1465],
                    [_ * 10**8 for _ in [9, 10, 11, 13, 16, 20, 24, 26, 27,
                                         27, 27]],
                    [0, -0.05],
                    [1, .99, .97, .95, .90, .85, .75, .65, .55, .40, .20],
                    [1, .99, .97, .95, .90, .85, .75, .65, .55, .40, .20],
                    [0, 0.1, 0.3, 0.5],
                    [0, 0.1, 0.3, 0.5]]
        x_to_2100 = np.linspace(1900, 2100, 11)
        x_0_to_100 = np.linspace(0, 100, 11)
        x_0_to_30 = np.linspace(0, 30, 4)
        x_values = [x_to_2100,
                    x_to_2100,
                    x_to_2100,
                    x_to_2100,
                    [0, 10],
                    x_0_to_100,
                    x_0_to_100,
                    x_0_to_30,
                    x_0_to_30]
        for func_name, x_vals, y_vals in zip(func_names, x_values, y_values):
            func = interp1d(x_vals, y_vals,
                            bounds_error=False,
                            fill_value=(y_vals[0],
                                        y_vals[-1]))
            setattr(self, func_name.lower()+"_f", func)
        # Delays
        var_dlinf3 = ["PCTI", "LMP"]
        for var_ in var_dlinf3:
            func_delay = Dlinf3(getattr(self, var_.lower()),
                                self.dt, self.time, method="euler")
            setattr(self, "dlinf3_"+var_.lower(), func_delay)

    def loopk_exogenous(self, k):
        """
        Run a sorted sequence to update one loop of the exogenous parameters.
        `@requires` decorator checks that all dependencies are computed
        previously.

        """
        j = k - 1
        kl = k
        jk = j

        self.pcti[k] = self.pcti[j] + self.dt * self.pctir[jk]

        self.pcrum[k] = self.pcrum_f(self.time[k])
        self.pop[k] = self.pop_f(self.time[k])
        self.aiph[k] = self.aiph_f(self.time[k])
        self.al[k] = self.al_f(self.time[k])

        #self.ppgf22[k] = self.dlinf3_pcti(k, self.tdd)
        #self.ppgf2 = switch(self.ppgf21, self.ppgf22[k], self.swat)

        self.plmp[k] = self.dlinf3_lmp(k, self.pd)
        self.pctcm[k] = self.pctcm_f(1 - self.plmp[k])

        self.pctir[kl] = clip(self.pcti[k] * self.pctcm[k], 0, self.time[k],
                              self.pyear)

        #self.lmp1[k] = self.lmp1_f(self.ppolx[k])
        #self.lmp2[k] = self.lmp2_f(self.ppolx[k])
        #self.lmp[k] = clip(self.lmp2[k], self.lmp1[k], self.time[k],
        #                  self.pyear)
        #self.lfdr1[k] = self.lfdr1_f(self.ppolx[k])
        #self.lfdr2[k] = self.lfdr1_f(self.ppolx[k])
        #self.lfdr[k] = clip(self.lfdr2[k], self.lfdr1[k], self.time[k],
        #                    self.pyear)

        self.io11[k] = .7e11*np.exp((self.time[k] - self.year_min)*.037)
        self.io12[k] = self.pop[k] * self.cio
        self.io1[k] = clip(self.io12[k], self.io11[k], self.time[k], self.lt2)
        self.io2[k] = .7e11 * np.exp(self.lt * .037)
        self.io[k] = clip(self.io2[k], self.io1[k], self.time[k], self.lt)
        self.iopc[k] = self.io[k] / self.pop[k]
        
        self.aiph[k] = (-((0.125*self.time[k])-250)**2)+160
        

    def loop0_exogenous(self):
        """
        Run a sequence to initialize the exogenous parameters (loop with k=0).

        """
        self.pcti[0] = 1

        self.pcrum[0] = self.pcrum_f(self.time[0])
        self.pop[0] = self.pop_f(self.time[0])
        self.aiph[0] = self.aiph_f(self.time[0])
        self.al[0] = self.al_f(self.time[0])

        #self.ppgf22[0] = self.dlinf3_pcti(0, self.tdd)
        #self.ppgf2 = switch(self.ppgf21, self.ppgf22[0], self.swat)

        self.lmp1[0] = self.lmp1_f(self.ppolx[0])
        self.lmp2[0] = self.lmp2_f(self.ppolx[0])
        self.lmp[0] = clip(self.lmp2[0], self.lmp1[0], self.time[0],
                           self.pyear)

        self.plmp[0] = self.dlinf3_lmp(0, self.pd)
        self.pctcm[0] = self.pctcm_f(1 - self.plmp[0])

        self.pctir[0] = clip(self.pcti[0] * self.pctcm[0], 0, self.time[0],
                             self.pyear)

        self.lfdr1[0] = self.lfdr1_f(self.ppolx[0])
        self.lfdr2[0] = self.lfdr1_f(self.ppolx[0])
        self.lfdr[0] = clip(self.lfdr2[0], self.lfdr1[0], self.time[0],
                            self.pyear)
        
        
        
    def loop0_pollution(self, alone=False):
        """
        Run a sequence to initialize the pollution sector (loop with k=0).

        Parameters
        ----------
        alone : boolean, optional
            if True, run the sector alone with exogenous inputs. The default
            is False.

        """

        self.pp[0] = self.pp19
        self.ppt[0] = 1 
        self._update_pcrum(0)
        self._update_ppolx(0)
        if alone:
            self.loopk_exogenous(0)
        self._update_ppgi(0)
        self._update_ppga(0)
        self._update_ppgf(0)
        self._update_ppgr(0)
        self._update_ppar(0)
        self._update_ahlm(0)
        self._update_ahl(0)
        self._update_ppasr(0)
        self._update_pptc(0)
        self._update_pptcm(0)
        self._update_pptcr(0,0)
        self._update_ppgf2(0)
        self._update_pptmi(0)
        self._update_pii(0)
        self._update_fio70(0)
        self._update_ymap1(0)
        self._update_ymap2(0)
        self._update_apfay(0)
        self._update_abl(0)
        self._update_ef(0)

    def loopk_pollution(self, j, k, jk, kl, alone=False):
        """
        Run a sequence to update one loop of the pollution sector.

        Parameters
        ----------
        alone : boolean, optional
            if True, run the sector alone with exogenous inputs. The default
            is False.

        """
        self._update_pcrum(k)
        
        self._update_ppolx(k)
        if alone:
            self.loopk_exogenous(k)
        self._update_ppgi(k)
        self._update_ppga(k)
        self._update_ppgf(k)
        self._update_ppgr(k)
        self._update_ppar(k)
        self._update_pp(k,j,jk)
        self._update_ahlm(k)
        self._update_ahl(k)
        self._update_ppasr(k)
        self._update_pptc(k)
        self._update_pptcm(k)
        self._update_pptcr(k,j)
        self._update_ppt(k,j)
        self._update_ppgf2(k)
        self._update_pptmi(k)
        self._update_pii(k)
        self._update_fio70(k)
        self._update_ymap1(k)
        self._update_ymap2(k)
        self._update_apfay(k)
        self._update_abl(k)
        self._update_ef(k)

    def run_pollution(self):
        """
        Run a sequence of updates to simulate the pollution sector alone with
        exogenous inputs.

        """
        self.redo_loop = True
        while self.redo_loop:
            self.redo_loop = False
            self.loop0_pollution(alone=True)

        for k_ in range(1, self.n):
            self.redo_loop = True
            while self.redo_loop:
                self.redo_loop = False
                if self.verbose:
                    print("go loop", k_)
                self.loopk_pollution(k_-1, k_, k_-1, k_, alone=True)

    @requires(["pcrum"])
    def _update_pcrum(self, k):
        """
        State variable, requires previous step only
        """
        
        self.pcrum[k] = self.pcrum_f(self.iopc[k])

    @requires(["ppgi"],["pcrum"])
    def _update_ppgi(self, k):
        """
        From step k requires: pcrum
        """

        self.ppgi[k] = self.pcrum[k]*self.pop[k]*self.frpm*self.imef*self.imti
    
    @requires(["ppga"],["aiph"])
    def _update_ppga(self, k):
        """
        From step k requires: aiph
        """

        self.ppga[k] = self.aiph[k]*self.al[k]*self.faipm*self.amti

    @requires(["ppgf"],["ppgf2"])
    def _update_ppgf(self, k):
        """
        From step k requires: nothing
        """
        
        self.ppgf[k] = clip(self.ppgf2[k], self.ppgf1, self.time[k], self.pyear_pp_tech)
        
    @requires(["ppgr"],["ppgf"],["ppga"],["ppgi"])
    def _update_ppgr(self, k):
        """
        From step k requires: ppgf, ppga, ppgi
        """
        
        self.ppgr[k] = (self.ppgi[k] + self.ppga[k]) * self.ppgf[k]

    @requires(["ppar"],["ppgr"])
    def _update_ppar(self, k):
        """
        From step k requires: ppgr
        """
        
        self.ppar[k] = self.dlinf3_ppgr(k, self.pptd)
        
    @requires(["pp"],["ppar", "ppasr"])
    def _update_pp(self, k,j,jk):
        """
        From step k requires: ppar, ppasr
        """

        self.pp[k] = self.pp[j] + self.dt*(self.ppar[jk] - self.ppasr[jk])

    @requires(["ppolx"],["pp"])
    def _update_ppolx(self, k):
        """
        From step k requires: pp
        """
        
        self.ppolx[k] = self.pp[k]/self.pp70

    @requires(["ahlm"],["ppolx"])
    def _update_ahlm(self, k):
        """
        From step k requires: ppolx
        """
        
        self.ahlm[k] = self.ahlm_f(self.ppolx[k])
    
    @requires(["ahl"],["ahlm"])
    def _update_ahl(self, k):
        """
        From step k requires: ahlm
        """
        
        self.ahl[k] = self.ahl70 * self.ahlm[k]

    @requires(["ppasr"],["ahl"])
    def _update_ppasr(self, k):
        """
        From step k requires: ahl
        """
        
        self.ppasr[k] = self.pp[k]/(1.4*self.ahl[k])

    @requires(["pptc"],["ppolx"])
    def _update_pptc(self, k):
        """
        From step k requires: ppolx
        """
        
        self.pptc[k] = 1-(self.ppolx[k]/self.dppolx)

    @requires(["pptcm"],["pptc"])
    def _update_pptcm(self, k):
        """
        From step k requires: pptc
        """
        
        self.pptcm[k] = self.pptcm_f(self.pptc[k]) 

    @requires(["pptcr"],["pptcm"])
    def _update_pptcr(self, k,j):
        """
        From step k requires: pptcm
        """
        
        if self.time[k] >= self.pyear_pp_tech:
            self.pptcr[k] = self.pptcm[j] * self.ppt[j]

        else:
            self.pptcr[k] = 0
        
    @requires(["ppt"],["pptcr"])
    def _update_ppt(self, k,j):
        """
        From step k requires: pptcr
        """

        self.ppt[k] = self.ppt[j] + self.dt*self.pptcr[k]

    @requires(["ppgf2"],["ppt"])
    def _update_ppgf2(self, k):
        """
        From step k requires: ppt
        """

        self.ppgf2[k] = self.dlinf3_ppt(k, self.tdt)
        
    @requires(["ppmi"],["ppgf"])
    def _update_pptmi(self, k):
        """
        From step k requires: ppgf
        """
        
        self.pptmi[k] =  self.pptmi_f(self.ppgf[k])

    @requires(["pii"],["ppgf","ppgi", "io"])
    def _update_pii(self, k):
        """
        From step k requires: ppgf, ppgi, io
        """
        
        self.pii[k] = self.ppgi[k] * self.ppgf[k] / self.io[k]

    @requires(["fio70"])
    def _update_fio70(self, k):
        """
        From step k requires: nothing
        """
        
        self.fio70[k] = self.io[k]/self.io70

    @requires(["ymap1"],["fio70"])
    def _update_ymap1(self, k):
        """
        From step k requires: fio70
        """
        
        self.ymap1[k] = self.ymap1_f(self.fio70[k])

    @requires(["ymap2"],["fio70"])
    def _update_ymap2(self, k):
        """
        From step k requires: fio70
        """
        
        self.ymap2[k] = self.ymap2_f(self.fio70[k])

    @requires(["apfay"],["ymap1","ymap2"])
    def _update_apfay(self, k):
        """
        From step k requires: ymap1, ymap2
        """
        
        if self.time[k] > self.apct:
            self.apfay[k] = self.ymap2[k]
     
        else:
            self.apfay[k] = self.ymap1[k]

    @requires(["abl"],["ppgr"])
    def _update_abl(self,k):
        """
        From step k requires: ppgr
        """
        
        self.abl[k] = self.ppgr[k] * self.ghup
        
    @requires (["ef"],["abl", "al", "uil"])
    def _update_ef(self,k):
        """
        From step k requires: abl, al, uil
        """
        
        self.ef[k] = (self.al[k]/1e9 + self.uil[k]/1e9 + self.abl[k])/1.91

