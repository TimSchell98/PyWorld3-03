import pyworld3.pollution as po
from pyworld3.utils import plot_world_variables


pol = po.Pollution()
pol.set_pollution_table_functions()
pol.init_pollution_variables()
pol.init_pollution_constants()
pol.set_pollution_delay_functions()
pol.init_exogenous_inputs()
pol.run_pollution()