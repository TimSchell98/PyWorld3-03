import pyworld3.agriculture as ag
import pyworld3.population as pop
import pyworld3.capital as cap

agr = ag.Agriculture()
agr.set_agriculture_table_functions()
agr.init_agriculture_constants()
agr.init_agriculture_variables()
agr.set_agriculture_delay_functions()
agr.init_exogenous_inputs()
agr.run_agriculture()
