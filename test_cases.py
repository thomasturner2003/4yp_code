import json
import model
import numpy as np
import time


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
        
file_paths = ['Dataset/expansion_triplets.json', 'Dataset/random_triplets.json', 'Dataset/contraction_triplets.json' ]
diameter_ratios= [1.1, 1, 0.9]
diameter = 10E-3
flow = model.Flow(5,998,1E-3,diameter)
solver = model.Solver("oriented", "turner", flow)

blasius_source = model.Data("blasius", flow)
for file_path, diameter_ratio in zip(file_paths, diameter_ratios):
    start = time.time()
    errors = []
    elbows_errors = []
    for case in json_case_generator(file_path):
        bends = [] 
        for r,o in zip(case['bend_radii'], case["twists"]):
            bends.append(model.Bend(int(r/diameter),o, diameter_ratio=diameter_ratio))
        pipes = []
        pipes.append(model.PipeSection(case['inlet_length']/diameter))
        for l in case['connector_lengths']:
            pipes.append(model.PipeSection(l/diameter))
        pipes.append(model.PipeSection(case['outlet_length']/diameter))
        calculated_pressure_drop = solver.get_pressure_drop(bends, pipes)
        cfd_pressure_drop = case['pressure_drop']
        error = (cfd_pressure_drop - calculated_pressure_drop) / cfd_pressure_drop
        errors.append(error)
        outlet_inlet_dp = blasius_source.get_pipe_loss(case['inlet_length'] + case['outlet_length'])
        elbows_errors.append((cfd_pressure_drop - calculated_pressure_drop) / (cfd_pressure_drop-outlet_inlet_dp))
    end = time.time()
    print(f"{file_path}, MEAN Absolute: {100*np.mean(np.abs(errors)):.2f}%, Excluding inlet and outlet: {100*np.mean(np.abs(elbows_errors)):.2f}%, num points: {len(errors)}, time taken: {(end-start):.2f}s")
    #print(f"{file_path}, MEDIAN Absolute: {100*np.median(np.abs(errors)):.2f}%, Excluding inlet and outlet: {100*np.median(np.abs(elbows_errors)):.2f}%, num points: {len(errors)}, time taken: {(end-start):.2f}s")
