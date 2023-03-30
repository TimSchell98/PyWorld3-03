import pyworld3.resource as re
from pyworld3.utils import plot_world_variables


rsc = re.Resource()
rsc.set_resource_table_functions()
rsc.init_resource_variables()
rsc.init_resource_constants()
rsc.set_resource_delay_functions()
rsc.init_exogenous_inputs()
rsc.run_resource()

plot_world_variables(rsc.time,
                     [rsc.nrfr, rsc.nr, rsc.pop],
                     ["NRFR", "NR", "POP"],
                     [[0, 1], [0, 2e12], [0, 16e9]],
                     figsize=(7, 5), title="Solo Resources")
