from mpi4py import MPI

from modules.binarypopulation import BinaryPopulation
from modules.step_SN import StepSN

from posydon.binary_evol.simulationproperties import SimulationProperties

from posydon.binary_evol.flow_chart import flow_chart
from posydon.binary_evol.MESA.step_mesa import CO_HeMS_step, MS_MS_step, CO_HMS_RLO_step
from posydon.binary_evol.DT.step_detached import detached_step
from posydon.binary_evol.CE.step_CEE import StepCEE


from posydon.binary_evol.DT.double_CO import DoubleCO
from posydon.binary_evol.step_end import step_end
from posydon.binary_evol.simulationproperties import TimingHooks, StepNamesHooks, PrintStepInfoHooks,EvolveHooks
import numpy as np
import pandas as pd
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument('--infile', type=str, required=True)
parser.add_argument('--outfile', type=str, required=True)
args = parser.parse_args()

class HotFix(EvolveHooks):
    """Add history column 'step_times' (time taken by step) to each binary.

    Example
    -------
    >>> pop.to_df(extra_columns=['step_times'])
    """

    def post_step(self, binary, step_name):
        if binary.state == 'disrupted':
            print(binary.event, binary.step_names)
            binary.star_2.log_LHe = -50
            binary.star_2.log_LHe_history[-1] = -50
            
            binary.star_2.log_LH = -50
            binary.star_2.log_LH_history[-1] = -50
            
            binary.star_2.center_c12 = 0
            binary.star_2.center_c12_history[-1] = 0
        return binary
            


oneline = pd.read_hdf(args.infile,'oneline')
number_of_binaries = len(oneline)
#indices = oneline.index.values

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

posydon_data_path = '/projects/b1119/POSYDON/data/POSYDON_data/'
# interp_method = 'linear_knn_interpolation'
interp_method = 'linear3c_kNN'
stop_method = 'stop_at_max_time'

sim_kwargs = dict(
    flow = flow_chart(),    

    step_HMS_HMS = (step_end,{}),
    step_CO_HeMS = (step_end,{}),
    step_CO_HMS_RLO = (step_end,{}),
    step_detached = (detached_step, dict(path=posydon_data_path)),
    step_CE = (StepCEE, {}),
    step_SN = (StepSN, {"max_Vkick": 500,
                        "max_BH_mass": 12.1}),
    step_dco = (step_end, {}),
    step_end = (step_end, {}),
    extra_hooks = [(TimingHooks, {}),(StepNamesHooks, {}),(PrintStepInfoHooks, {}),(HotFix,{})], # this is to include the column of 'step_times', 'step_names'
)


sim_prop = SimulationProperties(**sim_kwargs)
kwargs = {
          'file_name': filename,
          'number_of_binaries': number_of_binaries,
          'eccentricity_scheme':'zero',
          'optimize_ram': True,
          'extra_columns' : ['step_times','step_names'], 
          'only_select_columns' : ['state', 'event', 'time', 'lg_mtransfer_rate',
                                   'orbital_period', 'eccentricity', 'nearest_neighbour_distance','V_sys'],
#           'selection_function' : lambda binary: not(binary in pop.find_failed()),
          'include_S1' : True , 
          'S1_kwargs' : {'only_select_columns' : ['state', 'mass', 'metallicity', 
                                  'log_L', 'log_R', 'center_h1','center_he4',
                                  'he_core_mass', 'surface_he4', 'surface_h1'],
                         'scalar_names':['natal_kick_array']
          },
          'include_S2' : True,
          'S2_kwargs' : {'only_select_columns' : ['state', 'mass', 'metallicity', 
                                  'log_L', 'log_R', 'center_h1','center_he4',
                                  'he_core_mass', 'surface_he4', 'surface_h1'],
                         'scalar_names':['natal_kick_array']
           },
          'star_formation' : 'constant', # constant
          'max_simulation_time' : 1e10, # 10 Gyr
         }


binary_pop = BinaryPopulation(population_properties=sim_prop,**kwargs)

original_start_time = time.time()
print('loading steps')
sim_prop.load_steps()

start_time = time.time()
print('evolving')
binary_pop.evolve(from_hdf=True,verbose=True)

binary_pop.save(args.outfile, mode='w', **kwargs )

