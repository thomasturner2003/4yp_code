import json
import dsmiller
import numpy as np


def json_case_generator(file_path):
    """
    Loads the JSON file and yields one case (row) at a time
    as a dictionary.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # We use zip to iterate through all lists simultaneously
    # This effectively turns "columns" back into "rows"
    for b_radii, twists, i_len, o_len, c_lens, p_drop in zip(
        data['bend_radii'], 
        data['twists'], 
        data['inlet_length'], 
        data['outlet_length'], 
        data['connector_lengths'], 
        data['pressure_drop']
    ):
        # Create a dictionary for the single case
        case = {
            "bend_radii": b_radii,
            "twists": twists,
            "inlet_length": i_len,
            "outlet_length": o_len,
            "connector_lengths": c_lens,
            "pressure_drop": p_drop
        }
        yield case
        

file_paths = ['Dataset/long_long_triplets.json', 'Dataset/short_long_triplets.json', 'Dataset/short_short_triplets.json']
solver = dsmiller.Solver("oriented")
ito = dsmiller.Ito()
diameter = 10E-3
flow = dsmiller.Flow(5,998,1E-3,diameter)
errors = []
elbows_errors = []
for file_path in file_paths:
    for case in json_case_generator(file_path):
        bends = [] 
        for r,o in zip(case['bend_radii'], case["twists"]):
            bends.append(dsmiller.Bend(int(r/diameter),o,ito.get_k(r/diameter,flow.reynolds)))
        pipes = []
        pipes.append(dsmiller.PipeSection(case['inlet_length']/diameter))
        for l in case['connector_lengths']:
            pipes.append(dsmiller.PipeSection(l/diameter))
        pipes.append(dsmiller.PipeSection(case['outlet_length']/diameter))
        errors.append((case['pressure_drop'] - solver.get_pressure_drop(bends, pipes, flow)[0]) / case['pressure_drop'])
        outlet_inlet_dp = dsmiller.blasius_darcy(case['inlet_length']+ case['outlet_length'], flow)
        elbows_errors.append((case['pressure_drop'] - solver.get_pressure_drop(bends, pipes, flow)[0]) / (case['pressure_drop']-outlet_inlet_dp))
print(f"{100*np.mean(np.abs(errors)):.2f}%")
print("")
print(f"{100*np.mean(np.abs(elbows_errors)):.2f}%")
print("")
