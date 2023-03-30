# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from pyworld3 import World3
from pyworld3.utils import plot_world_variables

params = {'lines.linewidth': '3','axes.labelsize' : '12', 'xtick.labelsize' : '10', 'ytick.labelsize' : '10', 'figure.autolayout' : 'True'}
plt.rcParams.update(params)

"""
Choose Szenario:
    1: A Referenz Point
    2: More Abundant Nonrenewable Resources
    3: More Abundant Nonrenewable Resources and pollution control
    
Disclaimer: Szenario 2 and 3 do not match the szenarios of "Limits to Growth: The 30-year update", because some parameters were changed wich are not descriped.  
"""
szenario = 3

if szenario == 1:

    world3 = World3(dt = 1, pyear = 4000)
    world3.init_world3_constants()
    world3.init_world3_variables()
    world3.set_world3_table_functions()
    world3.set_world3_delay_functions()
    world3.run_world3(fast=False)

    plot_world_variables(world3.time,
                     [world3.nrfr, world3.io, world3.f, world3.pop,
                      world3.ppolx],
                     ["NRFR", "IO", "F", "POP", "PPOLX"],
                     [[0, 1.975], [0, 4e12], [0, 5.8e12], [0, 12e9], [0, 40]],
                     img_background="./img/fig 4-1-1.png",
                     figsize=(7, 5),
                     title="World3 Referenze Run, 2004 Szenario 1")
    
    plot_world_variables(world3.time,
                     [world3.le, world3.fpc, world3.sopc, world3.ciopc],
                     ["LE", "FPC", "SOPC", "CIOPC"],
                     [[0, 90], [0,1000],[0,970], [0, 250]],
                     img_background="./img/fig 4-1-2.png",
                     figsize=(7, 5),
                     title="World3 Referenze Run - Material standard of living, 2004 Szenario 1")
    
    
    plot_world_variables(world3.time,
                     [world3.ef, world3.hwi],
                     ["EF", "HWI"],
                     [[0, 4], [0,1]],
                     img_background="./img/fig 4-1-3.png",
                     figsize=(7, 5), title="World3 Referenze Run - Human Wellfare and Footprint, 2004 Szenario 1")
    
    print("Szenario 1, referenz run")
    
if szenario == 2:
    world3 = World3(dt = 1, pyear = 4000)
    world3.init_world3_constants(nri=2e12)
    world3.init_world3_variables()
    world3.set_world3_table_functions()
    world3.set_world3_delay_functions()
    world3.run_world3(fast=False)

    plot_world_variables(world3.time,
                     [world3.nrfr, world3.io, world3.f, world3.pop,
                      world3.ppolx],
                     ["NRFR", "IO", "F", "POP", "PPOLX"],
                     [[0, 0.9875], [0, 4e12], [0, 5.8e12], [0, 12e9], [0, 40]],
                     img_background="./img/fig 4-2-1.jpg",
                     figsize=(7, 5),
                     title="World3 More Resources, 2004 Szenario 2")

    plot_world_variables(world3.time,
                     [world3.le, world3.fpc, world3.sopc, world3.ciopc],
                     ["LE", "FPC", "SOPC", "CIOPC"],
                     [[0, 90], [0,1000],[0,970], [0, 250]],
                     img_background="./img/fig 4-2-2.jpg",
                     figsize=(7, 5),
                     title="World3 More Resources - Material standard of living, 2004 Szenario 2")

    plot_world_variables(world3.time,
                     [world3.ef, world3.hwi],
                     ["EF", "HWI"],
                     [[0, 4], [0,1]],
                     img_background="./img/fig 4-2-3.jpg",
                     figsize=(7, 5), title="World3 More Resources - Human Wellfare and Footprint, 2004 Szenario 2")
    
    print("Szenario 2: More Resources")
    
if szenario == 3:
    world3 = World3(dt = 1,pyear_pp_tech = 2002)
    world3.init_world3_constants(nri=2e12)
    world3.init_world3_variables()
    world3.set_world3_table_functions()
    world3.set_world3_delay_functions()
    world3.run_world3(fast=False)

    plot_world_variables(world3.time,
                     [world3.nrfr, world3.io, world3.f, world3.pop,
                      world3.ppolx],
                     ["NRFR", "IO", "F", "POP", "PPOLX"],
                     [[0, 1.975], [0, 4e12], [0, 6e12], [0, 12e9], [0, 40]],
                     img_background="./img/fig 4-3-1.jpg",
                     figsize=(7, 5),
                     title="World3 More Resources and Pollution Control, 2004 Szenario 2")

    plot_world_variables(world3.time,
                     [world3.le, world3.fpc, world3.sopc, world3.ciopc],
                     ["LE", "FPC", "SOPC", "CIOPC"],
                     [[0, 90], [0,1020],[0,970], [0, 250]],
                     img_background="./img/fig 4-3-2.jpg",
                     figsize=(7, 5),
                     title="World3 More Resources and Pollution Control - Material standard of living, 2004 Szenario 2")

    plot_world_variables(world3.time,
                     [world3.ef, world3.hwi],
                     ["EF", "HWI"],
                     [[0, 4.2], [0,1]],
                     img_background="./img/fig 4-3-3.jpg",
                     figsize=(7, 5), title="World3 More Resources and Pollution Control - Human Wellfare and Footprint, 2004 Szenario 2")
    
    print("Szenario 3: More Resources and Pollution Control")