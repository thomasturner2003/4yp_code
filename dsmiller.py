import pandas as pd
import numpy as np
import csv
import random
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
# GLOBAL


CORRECTION_FACTORS = {
    # (r/d = 1) + spacer + (r/d = 1)
    ('1', '1'): {
        0:   {0: 1.00, 30: 1.16, 60: 1.04, 90: 0.81, 120: 0.69, 150: 0.60, 180: 0.53},
        1:   {0: 0.86, 30: 1.04, 60: 0.93, 90: 0.79, 120: 0.69, 150: 0.63, 180: 0.58},
        4:   {0: 0.71, 30: 0.94, 60: 0.76, 90: 0.74, 120: 0.72, 150: 0.73, 180: 0.71},
        8:   {0: 0.81, 30: 0.83, 60: 0.82, 90: 0.82, 120: 0.81, 150: 0.91, 180: 0.80}
    },
    # (r/d = 1) + spacer + (r/d = 2)
    ('1', '2'): {
        0:   {0: 1.06, 30: 1.15, 60: 1.01, 90: 0.81, 120: 0.71, 150: 0.64, 180: 0.60},
        1:   {0: 0.91, 30: 1.05, 60: 0.96, 90: 0.86, 120: 0.78, 150: 0.71, 180: 0.64},
        4:   {0: 0.74, 30: 0.85, 60: 0.82, 90: 0.80, 120: 0.79, 150: 0.78, 180: 0.77},
        8:   {0: 0.83, 30: 0.84, 60: 0.84, 90: 0.82, 120: 0.81, 150: 0.81, 180: 0.81}
    },
    # (r/d = 1) + spacer + (r/d = 3)
    ('1', '3'): {
        0:   {0: 1.02, 30: 1.06, 60: 0.97, 90: 0.86, 120: 0.78, 150: 0.72, 180: 0.67},
        1:   {0: 0.93, 30: 1.05, 60: 0.92, 90: 0.90, 120: 0.82, 150: 0.78, 180: 0.72},
        4:   {0: 0.78, 30: 0.83, 60: 0.83, 90: 0.82, 120: 0.82, 150: 0.81, 180: 0.81},
        8:   {0: 0.87, 30: 0.83, 60: 0.83, 90: 0.83, 120: 0.83, 150: 0.84, 180: 0.85}
    },
    # (r/d = 2) + spacer + (r/d = 1)
    ('2', '1'): {
        0:   {0: 0.76, 30: 0.73, 60: 0.71, 90: 0.67, 120: 0.64, 150: 0.60, 180: 0.57},
        1:   {0: 0.74, 30: 0.72, 60: 0.70, 90: 0.68, 120: 0.66, 150: 0.64, 180: 0.62},
        4:   {0: 0.69, 30: 0.73, 60: 0.74, 90: 0.74, 120: 0.75, 150: 0.75, 180: 0.75},
        8:   {0: 0.74, 30: 0.79, 60: 0.80, 90: 0.80, 120: 0.81, 150: 0.81, 180: 0.81}
    },
    # (r/d = 2) + spacer + (r/d = 2)
    ('2', '2'): {
        0:   {0: 0.86, 30: 0.79, 60: 0.77, 90: 0.73, 120: 0.68, 150: 0.63, 180: 0.58},
        1:   {0: 0.83, 30: 0.79, 60: 0.74, 90: 0.71, 120: 0.68, 150: 0.65, 180: 0.62},
        4:   {0: 0.72, 30: 0.71, 60: 0.70, 90: 0.70, 120: 0.69, 150: 0.69, 180: 0.72},
        8:   {0: 0.77, 30: 0.81, 60: 0.81, 90: 0.80, 120: 0.80, 150: 0.80, 180: 0.80}
    },
    # (r/d = 2) + spacer + (r/d = 3)
    ('2', '3'): {
        0:   {0: 0.88, 30: 0.84, 60: 0.81, 90: 0.78, 120: 0.76, 150: 0.72, 180: 0.69},
        1:   {0: 0.85, 30: 0.83, 60: 0.81, 90: 0.79, 120: 0.77, 150: 0.75, 180: 0.73},
        4:   {0: 0.74, 30: 0.77, 60: 0.79, 90: 0.81, 120: 0.82, 150: 0.82, 180: 0.83},
        8:   {0: 0.83, 30: 0.82, 60: 0.83, 90: 0.84, 120: 0.85, 150: 0.86, 180: 0.87}
    },
    # (r/d = 3) + spacer + (r/d = 1)
    ('3', '1'): {
        0:   {0: 0.79, 30: 0.76, 60: 0.73, 90: 0.70, 120: 0.68, 150: 0.65, 180: 0.64},
        1:   {0: 0.76, 30: 0.75, 60: 0.73, 90: 0.72, 120: 0.70, 150: 0.69, 180: 0.68},
        4:   {0: 0.70, 30: 0.73, 60: 0.75, 90: 0.77, 120: 0.78, 150: 0.78, 180: 0.79},
        8:   {0: 0.76, 30: 0.81, 60: 0.81, 90: 0.81, 120: 0.82, 150: 0.82, 180: 0.82}
    },
    # (r/d = 3) + spacer + (r/d = 2)
    ('3', '2'): {
        0:   {0: 0.85, 30: 0.83, 60: 0.80, 90: 0.76, 120: 0.73, 150: 0.69, 180: 0.65},
        1:   {0: 0.83, 30: 0.81, 60: 0.79, 90: 0.76, 120: 0.74, 150: 0.71, 180: 0.68},
        4:   {0: 0.72, 30: 0.74, 60: 0.75, 90: 0.76, 120: 0.77, 150: 0.77, 180: 0.77},
        8:   {0: 0.77, 30: 0.81, 60: 0.80, 90: 0.80, 120: 0.80, 150: 0.80, 180: 0.80}
    },
    # (r/d = 3) + spacer + (r/d = 3)
    ('3', '3'): {
        0:   {0: 0.86, 30: 0.83, 60: 0.81, 90: 0.78, 120: 0.76, 150: 0.74, 180: 0.71},
        1:   {0: 0.87, 30: 0.85, 60: 0.83, 90: 0.79, 120: 0.77, 150: 0.75, 180: 0.73},
        4:   {0: 0.82, 30: 0.81, 60: 0.81, 90: 0.80, 120: 0.80, 150: 0.79, 180: 0.79},
        8:   {0: 0.85, 30: 0.85, 60: 0.85, 90: 0.85, 120: 0.85, 150: 0.85, 180: 0.85}
    }
}


# CLASSES


class PipeSection:
    """
    Length is in diameters
    """

    def __init__(self, length_d: float):
        self.length_d = length_d  # Ls/d: length in pipe diameters


class Bend:
    def __init__(self, r_d: int, orientation: float, k_value: float):
        """
        :param r_d: The relative radius of the bend (r/d). Must be 1, 2, or 3.
        :param orientation: The orientation of the bend in the global coordinate system (degrees).
        :param k_value: The single bend loss coefficient (K value).
        """
        self.r_d = r_d                  # Discrete bend radius (r/d, matching table values)
        self.orientation = orientation  # Orientation to the global system (degrees)
        self.k_value = k_value          # Loss coefficient (K value)
    
    
class Flow:
    def __init__(self, speed:float, rho:float, viscosiy:float, diameter:float):
        self.reynolds = rho*speed*diameter/viscosiy
        self.diameter =diameter
        self.speed = speed
        self.rho = rho
     
     
#METHODS   

 
def blasius_darcy(L:float, flow:Flow)->float:
    re = flow.reynolds
    f = 0.3164*(re**-0.25)
    return f*(L/flow.diameter)*(flow.rho*flow.speed**2)/2


def ito_k(curvature: int, re: float, theta:float=90)->float:
    """
    Calculates the K value of an elbow using the Ito method.
    
    Parameters:
    curvature: R/D
    re    : Reynolds number (dimensionless)
    theta : Bend angle (degrees)
    
    Returns:
    K     : Loss coefficient (dimensionless)
    """
    
    ratio = 2*curvature
    alpha = 0.95 + 17.2 * (ratio ** -1.96)
    
    # Calculate K (Equation 9)
    # K = 0.00241 * alpha * theta * (2R / D)^0.84 * Re^-0.17
    K = 0.00241 * alpha * theta * (ratio ** 0.84) * (re ** -0.17)
    return K


def relative_orientation(bend_1:Bend, bend_2:Bend)->int:
    o_1 = bend_1.orientation
    o_2 = bend_2.orientation
    diff = abs(o_2 - o_1)
    diff = min(diff, 360-diff)
    return diff
  
    
def get_bend_correction_factor(r_d_1: int, r_d_2: int, ls_d: int, theta_c: int) -> float:
    """
    Retrieves the interaction correction factor (C) for combinations of two 90° bends.

    Parameters:
    - r_d_1 (int): The relative radius (r/d) of the first 90° bend (1, 2, or 3).
    - r_d_2 (int): The relative radius (r/d) of the second 90° bend (1, 2, or 3).
    - ls_d (int): The spacer length ratio (Ls/d) between the bends (0, 1, 4, or 8).
    - theta_c (int): The combination angle (theta_c) in degrees (0, 30, 60, 90, 120, 150, or 180).

    Returns:
    - float: The interaction correction factor (C) from Table 10.1.
    - None: If the combination of parameters is not found in the table.
    """
    if ls_d > 8:
        return 1
    # Create the tuple key for the bend combination (ensure alphabetical order for lookup)
    # The table is structured such that (r/d = X) + spacer + (r/d = Y) is a distinct block.
    # Therefore, we treat the combination as-entered by the user.
    bend_key = (str(r_d_1), str(r_d_2))

    try:
        # Step 1: Lookup the outer dictionary (r/d combination)
        spacer_data = CORRECTION_FACTORS.get(bend_key)
        if not spacer_data:
            raise KeyError(f"r/d combination {r_d_1, r_d_2} not found.")

        # Step 2: Lookup the second level dictionary (Ls/d)
        theta_data = spacer_data.get(ls_d)
        if theta_data is None:
            raise KeyError(f"Ls/d value {ls_d} not found for combination {r_d_1, r_d_2}.")

        # Step 3: Lookup the final dictionary (theta_c)
        factor = theta_data.get(theta_c)
        if factor is None:
            raise KeyError(f"Theta_c value {theta_c} not found for combination {r_d_1, r_d_2} and Ls/d={ls_d}.")

        return factor

    except KeyError as e:
        print(f"Error: Parameter combination not found in table. {e}")
        return None


def calculate_pressure_drop(bends:list[Bend], pipes:list[PipeSection], flow:Flow, orientation_dependent:bool=True)->tuple[float,float,float]:
    """Calculates the pressure drop considering the level of shedding (scramble) from the previous inlet and the outlet correction only

    Args:
        bends (list[Bend]): Bends in pipe system
        pipes (list[PipeSection]): Straight sections in the pipe system
        flow (Flow): Flow in the pipe

    Raises:
        ValueError: Incorrect number of pipes or bends

    Returns:
        float: Total pressure drop [Pa]
    """
    # validity
    if  len(pipes) - len(bends) != 1:
            raise ValueError(f"You must have an inlet, outlet pipe and pipes between all elbows")
    # pressure drop in the pipes
    L_d = 0
    for pipe in pipes:
        L_d +=  pipe.length_d
    avg_drop = blasius_darcy(L_d*flow.diameter, flow)
    min_drop = avg_drop
    max_drop = avg_drop
    # pressure drop in the bends
    bend = bends[0]
    next_pipe = pipes[1]
    K_avg=find_scramble_k(bend.k_value, 1, flow.reynolds, bend.r_d, next_pipe.length_d)
    K_min = K_avg
    K_max = K_avg
    for i in range(1,len(bends)):
        prev_bend = bends[i-1]
        bend = bends[i]
        prev_pipe = pipes[i]
        next_pipe = pipes[i+1]
        if orientation_dependent:
            avg_scramble, min_scramble, max_scramble = get_scramble_coefficient(prev_bend.r_d, bend.r_d, prev_pipe.length_d,re=flow.reynolds, angles=[relative_orientation(prev_bend, bend)])
        else:
            avg_scramble, min_scramble, max_scramble = get_scramble_coefficient(prev_bend.r_d, bend.r_d, prev_pipe.length_d,re=flow.reynolds)
        K_avg+=find_scramble_k(bend.k_value, avg_scramble, flow.reynolds, bend.r_d, next_pipe.length_d)
        K_min+=find_scramble_k(bend.k_value, min_scramble, flow.reynolds, bend.r_d, next_pipe.length_d)
        K_max+=find_scramble_k(bend.k_value, max_scramble, flow.reynolds, bend.r_d, next_pipe.length_d)
    avg_drop += (0.5*flow.rho*flow.speed**2) * K_avg
    min_drop += (0.5*flow.rho*flow.speed**2) * K_min
    max_drop += (0.5*flow.rho*flow.speed**2) * K_max
    
    return avg_drop,min_drop,max_drop


def calculate_niave_drop(bends:list[Bend], pipes:list[PipeSection], flow:Flow)->float:
    # validity
    if  len(pipes) - len(bends) != 1:
            raise ValueError(f"You must have an inlet, outlet pipe and pipes between all elbows")
    # pressure drop in the pipes
    L_d = 0
    for pipe in pipes:
        L_d +=  pipe.length_d
    drop = blasius_darcy(L_d*flow.diameter, flow)
    # pressure drop in the bends
    K = 0
    for bend in bends:
        K+=bend.k_value
    drop += (0.5*flow.rho*flow.speed**2) * K
    return drop


def calculate_k(loss:float, L:float, flow:Flow)->float:
    """ Calculates the value of K given the loss

    Args:
        loss (float): Pressure drop [Pa]
        L (float): Pipe length [m]
        flow (Flow): Characteristic flow [-]

    Returns:
        float: K [-]
    """
    loss -= blasius_darcy(L, flow)
    return loss/(0.5*flow.rho*flow.speed**2)


def get_interpolated_correction_factor(r1, r2, separation, angle) -> float:
    """Finds the correction factor for an interpolated set of bends

    Args:
        r1 (_type_): R/D of bend 1 [-]
        r2 (_type_): R/D of bend 2 [-]
        separation (_type_): Seperation length/D [-]
        angle (_type_): Relative orientation [deg]

    Raises:
        ValueError: Does not interpolate between different curvatures will raise an error if data is not included
        
    Returns:
        float: Correction factor [-]
    """
    # Condition: If separation is 30 or greater, correction is exactly 1.0, this follows assumption from Miller
    if separation >= 30:
        return 1.0
    
    radii_key = (str(r1), str(r2))
    if radii_key not in CORRECTION_FACTORS:
        raise ValueError(f"Radii pair {radii_key} not found.")
    
    data_map = CORRECTION_FACTORS[radii_key]
    
    # Transform existing grid into log-space
    points = []
    values = []
    for sep, angles_dict in data_map.items():
        log_sep = np.log(1+sep)
        for ang, factor in angles_dict.items():
            points.append((log_sep, ang))
            values.append(factor)
            
    # Add the "Convergence Point" at sep=30 to the interpolation data and at sep=20 
    for ang in [0, 30, 60, 90, 120, 150, 180]:
        points.append((np.log(1+30), ang))
        values.append(1.0)
        points.append((np.log(1+20), ang))
        values.append(0.97)
            
    points = np.array(points)
    values = np.array(values)
    
    # Interpolate target in log-space
    target_point = np.array([[np.log1p(separation), angle]])
    result = griddata(points, values, target_point, method='linear')[0]
    
    # Fallback to nearest if on the very edge
    if np.isnan(result):
        result = griddata(points, values, target_point, method='nearest')[0]
        
    return float(result)


def interpolate_re_correction(re: float, input_filename="reynolds correction.csv") -> float:
    """Interpolates the correction factor for the given reynolds from data from DS Miller

    Args:
        re (float): Reynolds number [-]
        input_filename (str, optional): _description_. Defaults to "reynolds correction.csv".

    Returns:
        float: Correction factor [-]
    """
    # 1. Load data from CSV
    df = pd.read_csv(input_filename)
    re_original = df['Re'].values
    c_original = df['C'].values
    
    # 2. Transform original Re and input Re to log-space
    log_re_orig = np.log10(re_original)
    log_re_input = np.log10(re)
    
    # 3. Interpolate C based on log10(Re)
    # points: (N, 1) array of known log(Re) coordinates
    # xi: (1, 1) array of the target log(Re) coordinate
    points = log_re_orig.reshape(-1, 1)
    xi = np.array([[log_re_input]])
    
    # griddata returns an array, so we take the first element [0]
    c_value = griddata(points, c_original, xi, method='linear')[0][0]
    
    return float(c_value)


def interpolate_k_outlet_correction(k_star: float, outlet_length: float, input_filename="outlet correction.csv") -> float:
    """Calculates the correction factor due to shortened outlet using data from DS Miller

    Args:
        k_star (float): K value at Reynolds = 1E6 [-]
        outlet_length (float): length/D [-]
        input_filename (str, optional): _description_. Defaults to "outlet correction.csv".

    Returns:
        float: Correction factor [-]
    """
    # 1. Load data from CSV
    df = pd.read_csv(input_filename)
    k_vals = df['K_star'].values
    len_vals = df['Outlet length'].values
    c_vals = df['C'].values
    
    # 2. Natural log transformation with +1 offset
    ln_len_orig = np.log(len_vals + 1)
    ln_len_input = np.log(outlet_length + 1)
    
    # 3. Define the coordinate space (K*, ln(L+1))
    points = np.column_stack((k_vals, ln_len_orig))
    xi = np.array([[k_star, ln_len_input]])
    
    # 4. Perform Bilinear (Linear 2D) Interpolation
    # Returns the interpolated C factor for the given (k_star, outlet_length)
    c_value = griddata(points, c_vals, xi, method='linear')[0]
    
    return float(c_value)


def calculate_scramble_coefficient(k1_star:float, k2_star:float, correction:float, seperation:float)->float:
    """Calculates the scramble coefficient for a set of bends

    Args:
        k1_star (float): first bend K value at Re 1E6 [-]
        k2_star (float): second bend K value at Re 1E6 [-]
        correction (float): correction for the bend combination given by DS Miller [-]
        seperation (float): seperation distance/D [-]
        
    Returns:
        float: scramble coefficient [-]
    """
    return (correction*(k1_star+k2_star) - k1_star*interpolate_k_outlet_correction(k1_star, seperation))/k2_star


def get_scramble_coefficient(rd1:float, rd2:float, sep:float, angles=[0,30,60,90,120,150,180], re=50E3)->tuple[float,float,float]:
    """ Calculates the average, minimum, and maximum scramble coefficients across a set of orientations.

    Args:
        rd1 (float): R/D for bend 1 [-]
        rd2 (float): R/D for bend 2 [-]
        sep (float): seperation/D [-]
        angles (list, optional): relative orientations to test [deg]. Defaults to [0,30,60,90,120,150,180].
        re (_type_, optional): Reynolds number [-]. Defaults to 50E3.

    Returns:
        tuple[float,float,float]: avergae, minimum, maximum scramble coefficients [-,-,-]
    """
    angle_scrambles = []
    
    # Pre-calculate constant values for this R/D and Reynolds number
    k1_star = ito_k(rd1, re) / interpolate_re_correction(re)
    k2_star = ito_k(rd2, re) / interpolate_re_correction(re)
    
    for angle in angles:
        # Get correction factor for this specific orientation
        cf = get_interpolated_correction_factor(rd1, rd2, sep, angle)
        
        # Calculate individual scramble
        s = calculate_scramble_coefficient(k1_star, k2_star, cf, sep)
        angle_scrambles.append(s)
    
    # Calculate statistics
    avg_scramble = sum(angle_scrambles) / len(angle_scrambles)
    min_scramble = min(angle_scrambles)
    max_scramble = max(angle_scrambles)
    
    return avg_scramble, min_scramble, max_scramble


def find_scramble_k(k:float, scramble_coefficient:float, re:float, curvature:float, outflow_length:float)->float:
    """Finds the K using the scramble and the outlet correction

    Args:
        k (float): Pressure loss coefficent [-]
        re (float): Reynolds number [-]
        curvature (float): Radius of curvature/Diameter [-]
        inflow_length (float): Inflow length/Diameter [-]
        outflow_length (float): Outflow length/Diameter [-]

    Raises:
        ValueError: If curvature is not in the data set

    Returns:
        float: k
    """
    k_star = k/interpolate_re_correction(re)
    outlet_correction = interpolate_k_outlet_correction(k_star,outflow_length)
    if  curvature not in [2,3]:
            raise ValueError(f"You must give a power law relationship for the curvature")
    
    return k*outlet_correction*scramble_coefficient


#Tests


def run_sweep(v):
    filename = "combinations_1D_results.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["bend_radius", "bend_radius", "orientation", "seperation", "miller_value", "niave_value"])
        orientations = [0, 90, 180]
        seperations = [0,1,4,8]
        bend_radii =[2,3]
        k_vals = [ito_k(2, v*10E3), ito_k(3,v*10E3)]
        inlet = PipeSection(5)
        outlet = PipeSection(40)
        flow = Flow(v,998,1E-3,0.01)
        for i in [0,1]:
            bend_radius_1 = bend_radii[i]
            k_1 = k_vals[i]
            for l in [0,1]:
                bend_radius_2 = bend_radii[l]
                if bend_radius_1 != bend_radius_2:
                    continue
                k_2 = k_vals[l]
                for j in [0,1,2,3]:
                    seperation = seperations[j]
                    for z in [0,1,2]:
                        orientation = orientations[z]
                        bend1 = Bend(bend_radius_1,0,k_1)
                        connector1 = PipeSection(seperation)
                        bend2 = Bend(bend_radius_2,orientation,k_2)
                        bends = [bend1, bend2]
                        pipes = [inlet, connector1, outlet]
                        true_value = calculate_pressure_drop(bends, pipes, flow)
                        niave_value = calculate_niave_drop(bends, pipes, flow)
                        writer.writerow([bend_radius_1, bend_radius_2, orientation, seperation, true_value, niave_value])
    print("executed")
    return


def sample_and_export_to_csv(filename="interpolation_samples.csv", n_samples=100, seed=42):
    random.seed(seed)
    
    # Setup discrete sampling sets
    angles_set = [0, 90, 180]
    radii_set = [2, 3]
    # Specific separation values as requested
    separation_set = [0.5, 1.5, 2, 3, 5, 6, 7, 9, 10, 15, 20, 25]
    
    data_rows = []
    
    for _ in range(n_samples):
        # Randomly select from the provided sets
        r_d = random.choice(radii_set)
        orientation = random.choice(angles_set)
        separation = random.choice(separation_set)
        
        # Calculate factor (assumes r1=r2 based on "singular" requirement)
        factor = get_interpolated_correction_factor(r_d, r_d, separation, orientation)
        
        data_rows.append({
            'r/d': r_d,
            'orientation': orientation,
            'separation': separation,
            'correction_factor': round(factor, 4)
        })
    
    # Write to CSV using built-in library
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['r/d', 'orientation', 'separation', 'correction_factor'])
        writer.writeheader()
        writer.writerows(data_rows)
    
    print(f"Successfully exported {n_samples} samples to {filename}")


def run_loocv_validation():
    all_errors = []
    threshold = 0.1
    
    for radii_pair in list(CORRECTION_FACTORS.keys()):
        r1, r2 = radii_pair
        
        # Skip r/d = 1 cases
        if r1 == '1' or r2 == '1':
            continue
        for sep in list(CORRECTION_FACTORS[radii_pair].keys()):
            if sep == 0:
                    continue
            for ang in list(CORRECTION_FACTORS[radii_pair][sep].keys()):
                if ang == 0 or ang == 180:
                    continue
                actual_val = CORRECTION_FACTORS[radii_pair][sep][ang]
                del CORRECTION_FACTORS[radii_pair][sep][ang]
                
                try:
                    # Predicted via the log-interpolator
                    predicted = get_interpolated_correction_factor(r1, r2, sep, ang)
                    error = abs(predicted - actual_val)/actual_val
                    all_errors.append(error)
                    
                    if error > threshold:
                        print(f"High Error | Radii: {radii_pair} | Sep: {sep} | Ang: {ang} | Error: {error:.4f}")
                    
                finally:
                    # Restore global data
                    CORRECTION_FACTORS[radii_pair][sep][ang] = actual_val

    all_errors = np.array(all_errors)
    if len(all_errors) == 0: return 0.0, 0.0, 0.0
        
    return np.mean(all_errors), np.max(all_errors), np.sum(all_errors > threshold) / len(all_errors)


def power_law():
    r_d = [(3,2), (2,3)]
    seperation = range(4, 16)
    angles = [0, 30, 60, 90, 120, 150, 180]
    re = 50E3
    plt.figure(figsize=(12, 7))

    for rd1, rd2 in r_d:
        scramble_avgs = []
        ln_sep_list = []
        lower_err = []
        upper_err = []
        
        for sep in seperation:
            angle_scrambles = []
            for angle in angles:
                cf = get_interpolated_correction_factor(rd1, rd2, sep, angle)
                s = calculate_scramble_coefficient(
                    ito_k(rd1, re) / interpolate_re_correction(re), 
                    ito_k(rd2, re) / interpolate_re_correction(re), 
                    cf, 
                    sep
                )
                angle_scrambles.append(s)
            
            avg_s = np.mean(angle_scrambles)
            scramble_avgs.append(avg_s)
            ln_sep_list.append(np.log(sep))
            
            # Use max(0, ...) to prevent tiny negative floating point errors
            lower_err.append(max(0, avg_s - min(angle_scrambles)))
            upper_err.append(max(0, max(angle_scrambles) - avg_s))

        # Reformat for matplotlib errorbar
        yerr = [lower_err, upper_err]

        # --- Power Law Fit ---
        ln_scramble = np.log(scramble_avgs)
        b, ln_a = np.polyfit(ln_sep_list, ln_scramble, 1)
        a = np.exp(ln_a)
        
        print(f"R/D {rd1}, {rd2} | Power Law: Scramble = {a:.4f} * Separation^{b:.4f}")

        # Plot original data with error bars
        line, caps, bars = plt.errorbar(ln_sep_list, scramble_avgs, yerr=yerr, fmt='o', 
                                        capsize=4, label=f'Data R/D={rd1}, {rd2}', markersize=5)
        
        # Plot fit line using the same color as the error bar plot
        fit_scramble = a * np.exp(ln_sep_list)**b
        plt.plot(ln_sep_list, fit_scramble, '--', color=line.get_color(), 
                 alpha=0.6, label=f'Fit (b={b:.2f})')

    plt.xlabel('ln(Separation)')
    plt.ylabel('Scramble')
    plt.title('Power Law Fit with Orientation Min/Max Spread')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.show()


for sep1, sep2 in [(1,1),(2,2), (3,3), (4,4)]:
    k1 = ito_k(3,50E3)
    k2 = ito_k(2,50E3)
    bend_rd2 = Bend(2, 0, k2)
    bend_rd3 = Bend(3, 0, k1)
    flow = Flow(5,998,1E-3,10E-3)
    inlet = PipeSection(5)
    bend1 = bend_rd2
    connector1 = PipeSection(sep1)
    bend2 = bend_rd2
    connector2 = PipeSection(sep2)
    bend3 = bend_rd2
    outlet = PipeSection(40)
    bends = [bend1, bend2, bend3]
    pipes = [inlet, connector1, connector2, outlet]
    print(calculate_pressure_drop(bends, pipes, flow, orientation_dependent=True)[0])
