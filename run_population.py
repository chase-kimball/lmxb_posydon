from mpi4py import MPI

from posydon.popsyn.binarypopulation import BinaryPopulation
from posydon.binary_evol.simulationproperties import SimulationProperties

from posydon.binary_evol.flow_chart import flow_chart
from posydon.binary_evol.MESA.step_mesa import CO_HeMS_step, MS_MS_step, CO_HMS_RLO_step
from posydon.binary_evol.DT.step_detached import detached_step
from posydon.binary_evol.CE.step_CEE import StepCEE
from posydon.binary_evol.SN.step_SN import StepSN
from posydon.binary_evol.DT.double_CO import DoubleCO
from posydon.binary_evol.step_end import step_end
from posydon.binary_evol.simulationproperties import TimingHooks, StepNamesHooks
import numpy as np
import pandas as pd
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--m1min', type=float, default=5.0)
parser.add_argument('--m1max', type=float, default=120.0)
parser.add_argument('--m2min', type=float, default=0.2)
parser.add_argument('--m2max', type=float, default=3.0)
parser.add_argument('--Nbin', type=int, default=100_000)
parser.add_argument('--Amin', type=float, default=5.0)
parser.add_argument('--Amax', type=float, default=1e4)
parser.add_argument('--outfile', type=str, required=True)
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

posydon_data_path = '/projects/b1119/POSYDON/data/POSYDON_data/'
# interp_method = 'linear_knn_interpolation'
interp_method = 'linear3c_kNN'
stop_method = 'stop_at_max_time'

sim_kwargs = dict(
    flow = flow_chart(),    
#     step_HMS_HMS = (step_end, {}),
#     step_CO_HeMS = (step_end, {}),
#     step_CO_HMS_RLO = (step_end, {}),
    step_HMS_HMS = (MS_MS_step, dict(interpolation_method=interp_method,
                                     stop_method=stop_method,
                                     path=posydon_data_path,
                                     interpolation_path='/projects/b1119/POSYDON_POPSYNTH/20211206/',
                                     interpolation_filename='linear3c_kNN_q_neq_1.pkl')),
    step_CO_HeMS = (CO_HeMS_step, dict(interpolation_method=interp_method,
                                       stop_method=stop_method, 
                                       path=posydon_data_path,
                                       interpolation_path=posydon_data_path+'CO-HeMS/interpolators/linear3c_kNN/',
                                       interpolation_filename='grid_0.0142.pkl')),
    step_CO_HMS_RLO = (CO_HMS_RLO_step, dict(interpolation_method=interp_method,
                                             stop_method=stop_method,
                                             path=posydon_data_path,
                                             interpolation_path=posydon_data_path+'CO-HMS_RLO/interpolators/linear3c_kNN/',
                                             interpolation_filename='grid_0.0142.pkl')),
    step_detached = (detached_step, dict(path=posydon_data_path)),
    # step_HMS_detached = (step_end, {}), #(HMS_detached_step, {}), 
    # step_HeMS_detached = (step_end, {}), #(HeMS_detached_step, {}), 
    step_CE = (StepCEE, {}),
    step_SN = (StepSN, {"use_interp_values": True,
                        "use_profiles": True,
                        "use_core_masses": True}),
    step_dco = (DoubleCO, {}),
    step_end = (step_end, {}),
    extra_hooks = [(TimingHooks, {}),(StepNamesHooks, {})], # this is to include the column of 'step_times', 'step_names'
)


sim_prop = SimulationProperties(**sim_kwargs)
kwargs = {'number_of_binaries' : args.Nbin,
          'primary_mass_min' : args.m1min,
          'primary_mass_max' : args.m1max,
          'secondary_mass_scheme' : 'flat_mass_ratio',
          'secondary_mass_min': args.m2min,
          'secondary_mass_max': args.m2max, 
          'orbital_separation_min': args.Amin,
          'orbital_separation_max': args.Amax, #cut from the bottom
          'eccentricity_scheme':'zero',
          'extra_columns' : ['step_times','step_names'], 
          'only_select_columns' : ['state', 'event', 'time', 'lg_mtransfer_rate',
                                   'orbital_period', 'eccentricity', 'nearest_neighbour_distance'],
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

comm = MPI.COMM_WORLD
binary_pop = BinaryPopulation(population_properties=sim_prop,  comm=comm, **kwargs)

original_start_time = time.time()
sim_prop.load_steps()
if rank == 0:
    print(f'rank {rank} loaded all steps in t={(time.time()-original_start_time)/60:.2f} min')

start_time = time.time()
binary_pop.evolve()
if rank == 0:
    print(f'rank {rank} done evolving {len(binary_pop.manager.history_dfs)} binaries in t={time.time() - start_time:.2f}')

binary_pop.save( args.outfile, **kwargs )
if rank == 0:
    print('rank:', rank, 'Run time:', time.time() - original_start_time)
