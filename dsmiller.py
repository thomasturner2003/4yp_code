import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import os
import json


# CLASSES


class PipeSection:
    """
    Length is in diameters
    """
    def __init__(self, length_d: float):
        self.length_d = length_d


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
     
   
class Solver:
    supported_solvers = ["isolated", "orientationless", "oriented"]
    def __init__(self, solver_type:str):
        if solver_type.lower() not in self.supported_solvers:
            raise ValueError(f"{solver_type} not in supported solvers: {self.supported_solvers}")
        self.solver_type = solver_type.lower()  
    
    def _interacting_pressure_drop(self, bends:list[Bend], pipes:list[PipeSection], flow:Flow, orientation_dependent:bool=True)->tuple[float,float,float]:
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
                avg_scramble, min_scramble, max_scramble = get_scramble_coefficient(prev_bend.r_d, bend.r_d, prev_pipe.length_d,re=flow.reynolds, angles=[self.relative_orientation(prev_bend, bend)])
            else:
                avg_scramble, min_scramble, max_scramble = get_scramble_coefficient(prev_bend.r_d, bend.r_d, prev_pipe.length_d,re=flow.reynolds)
            K_avg+=find_scramble_k(bend.k_value, avg_scramble, flow.reynolds, bend.r_d, next_pipe.length_d)
            K_min+=find_scramble_k(bend.k_value, min_scramble, flow.reynolds, bend.r_d, next_pipe.length_d)
            K_max+=find_scramble_k(bend.k_value, max_scramble, flow.reynolds, bend.r_d, next_pipe.length_d)
        avg_drop += (0.5*flow.rho*flow.speed**2) * K_avg
        min_drop += (0.5*flow.rho*flow.speed**2) * K_min
        max_drop += (0.5*flow.rho*flow.speed**2) * K_max
        return avg_drop,min_drop,max_drop

    def _isolated_pressure_drop(self, bends:list[Bend], pipes:list[PipeSection], flow:Flow)->float:
        """Calculates the pressure drop assuming all bends are in isolation

        Args:
            bends (list[Bend]):
            pipes (list[PipeSection]):
            flow (Flow): 

        Raises:
            ValueError: Incorrect number of pipes relative to bends

        Returns:
            float: pressure drop [Pa]
        """
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
    
    def get_pressure_drop(self, bends:list[Bend], pipes:list[PipeSection], flow:Flow)->tuple[float,float,float]:
        if not self.solver_type:
            raise RuntimeError(f"Solver not set")
        if self.solver_type == "isolated":
            pd = self._isolated_pressure_drop(bends, pipes, flow)
            return pd,pd,pd
        elif self.solver_type == "orientationless":
            return self._interacting_pressure_drop(bends, pipes, flow, False)
        elif self.solver_type == "oriented":
            return self._interacting_pressure_drop(bends, pipes, flow, True)
        else:
            print("solver not found")
            return -1
    
    def relative_orientation(bend_1:Bend, bend_2:Bend)->float:
        """Calculates the relative orientation of 2 bends

        Args:
            bend_1 (Bend):
            bend_2 (Bend): 

        Returns:
            float: relative orientation [deg]
        """
        o_1 = bend_1.orientation
        o_2 = bend_2.orientation
        diff = abs(o_2 - o_1)
        diff = min(diff, 360-diff)
        return diff
  

class Data:
    supported_sources = ["turner", "miller", "ito", "blasius"]
    supported_curvatures = [2,3]
    
    def __init__(self, source:str, flow:Flow):
        if source.lower() not in self.supported_sources:
            raise RuntimeError(f"No data for {source} in supported sources: {self.supported_sources}")
        self.source = source
        self.flow = flow
        
    # Find the pipe pressure loss
    def get_pipe_loss(self, L:float)->float:
        if self.source != "blasius":
            raise RuntimeError("Only Blasius data supports pipe loss")
        return self._blasius_darcy(L)
    
    def _blasius_darcy(self, L:float)->float:
        """Calculates the pressure drop givent the Blasius relation for the Darcy friction factor

        Args:
            L (float): Length of pipe [m]
            flow (Flow): Flow characteristic [-]

        Returns:
            float: Pressure drop [Pa]
        """
        re = self.flow.reynolds
        f = 0.3164*(re**-0.25)
        return f*(L/self.flow.diameter)*(self.flow.rho*self.flow.speed**2)/2

    # Finding K for elbow
    def get_elbow_k(self, rd:int, diameter_ratio:float=1)->float:
        if self.source == "ito":
            if diameter_ratio != 1:
                raise ValueError("diameter ratio in Ito method must be 1")
            return self._ito_k(rd, self.flow.reynolds)
        elif self.source == "turner":
            return self._turner_elbow_k(rd, self.flow.reynolds, diameter_ratio)
        else:
            raise RuntimeError("Source not supported for get_elbow_k")
            
    def _ito_k(self, rd: int, re: float, theta:float=90)->float:
        """
        Calculates the K value of an elbow using the Ito method.
        
        Parameters:
        curvature: R/D [-]
        re    : Reynolds number [-]
        theta : Bend angle [deg]
        
        Returns:
        K     : Loss coefficient [-]
        """
        
        ratio = 2*rd
        alpha = 0.95 + 17.2 * (ratio ** -1.96)
        
        # K = 0.00241 * alpha * theta * (2R / D)^0.84 * Re^-0.17
        K = 0.00241 * alpha * theta * (ratio ** 0.84) * (re ** -0.17)
        return K

    def _turner_elbow_k(self, rd:int, re:float, diameter_ratio:float)->float:
        diameter_ratios = [0.9, 1, 1.1]
        if abs(re-50E3) >1000:
            raise ValueError("For Turner method the Reynolds number needs to ~ 50,000")
        if diameter_ratio not in diameter_ratios:
            raise ValueError(f"Turner method only supports diameter ratios {diameter_ratios}")
        with open('Data Sources/turner_elbow_k.json', 'r') as f:
            lookup_data = json.load(f)
        for entry in lookup_data:
            if abs(entry["diameter_ratio"] - diameter_ratio) < 1e-6 and abs(entry["R/D"] - rd) < 1e-6:
                return entry["K"]
        raise ValueError(f"No match found in Turner elbow diameter for diameter_ratio {diameter_ratio} and R/D {rd}")
         
    # Finding correction factors for elbows
    def get_elbow_correction_factor(self, rd1:int, rd2:int, seperation:float, twist:float)->float:
        """
        Retrieves the interaction correction factor (C) for combinations of two 90° bends.

        Parameters:
        - rd1 (int): The relative radius (r/d) of the first 90° bend.
        - rd2 (int): The relative radius (r/d) of the second 90° bend.
        - seperation (float): The spacer length ratio (Ls/d) between the bends (0, 1, 4, or 8).
        - twist (float)): The twist angle (theta_c) in degrees (0, 30, 60, 90, 120, 150, or 180).

        Returns:
        - float: The interaction correction factor
        """
        if self.source != "miller":
            raise RuntimeWarning("Data source for elbow correction factor should be Miller")
        if rd1 not in self.supported_curvatures or rd2 not in self.supported_curvatures:
            raise ValueError(f"Curvature not in supported curvatures {self.supported_curvatures}")
        return self._miller_interpolated_elbow_correction(rd1,rd2,seperation,twist)

    def _miller_interpolated_elbow_correction(self, rd1: int, rd2: int, seperation: float, twist: float, json_path="Data Sources/miller_elbow_correction_factors.json") -> float:
        
        # Condition: If separation is 30 or greater, correction is exactly 1.0
        if seperation >= 30:
            return 1.0
        
        # 1. Load the data directly from JSON
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Could not find {json_path}")

        with open(json_path, 'r') as f:
            full_data = json.load(f)
            # Assuming the JSON root is {"correction_factors": {...}}
            correction_factors = full_data.get("correction_factors", {})

        # 2. Match the JSON key format "r1-r2"
        radii_key = f"{int(rd1)}-{int(rd2)}"
        
        if radii_key not in correction_factors:
            raise ValueError(f"Radii pair {radii_key} not found in {json_path}.")
        
        data_map = correction_factors[radii_key]
        
        # 3. Transform existing grid into log-space
        points = []
        values = []
        
        for sep_str, angles_dict in data_map.items():
            # JSON keys are strings, cast back to float for log math
            log_sep = np.log1p(float(sep_str)) 
            
            for ang_str, factor in angles_dict.items():
                points.append((log_sep, float(ang_str)))
                values.append(factor)
                    
        # 4. Add Convergence Points (Miller assumptions)
        # Adding points at sep=30 (factor 1.0) and sep=20 (factor 0.97)
        for ang in [0, 30, 60, 90, 120, 150, 180]:
            points.append((np.log1p(30), float(ang)))
            values.append(1.0)
            points.append((np.log1p(20), float(ang)))
            values.append(0.97)
                    
        points = np.array(points)
        values = np.array(values)
        
        # 5. Interpolate target in log-space
        target_point = np.array([[np.log1p(seperation), twist]])
        
        # Linear interpolation
        result = griddata(points, values, target_point, method='linear')[0]
        
        # Fallback to nearest if on the very edge (NaN protection)
        if np.isnan(result):
            result = griddata(points, values, target_point, method='nearest')[0]
            
        return float(result)
    
    # Correcting for Reynold's correction
    def get_reynolds_correction_factor(self)->float:
        if self.source != "miller":
            raise RuntimeWarning("Data source for Reynolds correction should be Miller")
        return self._miller_interpolated_reynolds_correction(flow.reynolds)
    
    def _miller_interpolated_reynolds_correction(self,re:float,json_path="Data Sources/miller_reynolds_correction.json")->float:
        # 1. Load data from JSON
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Could not find {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        # 2. Extract Re and C values from the list of objects format
        # Using the "List of Objects" format provided in the previous step
        re_original = np.array([item["Re"] for item in data])
        c_original = np.array([item["C"] for item in data])
        
        # 3. Handle out-of-bounds cases (Optional but recommended)
        if re <= re_original.min():
            return float(c_original[0])
        if re >= re_original.max():
            return float(c_original[-1])

        # 4. Transform original Re and input Re to log-space
        log_re_orig = np.log10(re_original)
        log_re_input = np.log10(re)
        
        # 5. Interpolate C based on log10(Re)
        points = log_re_orig.reshape(-1, 1)
        xi = np.array([[log_re_input]])
        
        # griddata returns an array; we extract the scalar value
        c_value = griddata(points, c_original, xi, method='linear')[0]
        
        # Fallback to nearest if linear fails on the boundary
        if np.isnan(c_value):
            c_value = griddata(points, c_original, xi, method='nearest')[0]
        
        return float(c_value)
    
    # Correcting for shortened outlet
    def get_outlet_correction_factor(self, rd:int, outlet_length:float, diameter_ratio:float=1)->float:
        if rd not in self.supported_curvatures:
            raise ValueError("Curvature not supported")
        
        if self.source == "miller":
            if diameter_ratio != 1:
                raise ValueError("Miller data for outlet correction only supports area ratio of 1")
            return self._miller_outlet_correction_factor(rd, outlet_length)
        elif self.source == "turner":
            return self._turner_outlet_correction_factor(rd,diameter_ratio,outlet_length)
        else:
            raise RuntimeWarning("Data source for outlet correction factor should be Miller or Turner")
        
    def _turner_outlet_correction_factor(self, rd:int, diameter_ratio:float, outlet_length:float, input_filename="Data Sources/turner_elbow_correction_factors.json")->float:
        """
        Calculates correction factor by filtering JSON for diameter_ratio and R/D.
        
        Args:
            diameter_ratio (float): The ratio of diameters (e.g., 0.9, 1.0, 1.1)
            rd (float): The R/D ratio (e.g., 2, 3)
            outlet_length (float): The length/D to interpolate for.
        """
        # 1. Load the JSON list
        with open(input_filename, 'r') as f:
            data_list = json.load(f)
        # 2. Filter by R/D and prepare points for griddata
        points = []  # Will hold pairs of (diameter_ratio, ln_outlet_length)
        values = []  # Will hold the corresponding C factors
        
        for entry in data_list:
            # Filter by the specific R/D curve
            if abs(entry['R/D'] - rd) < 1e-6:
                dr = entry['diameter_ratio']
                
                for len_str, c_val in entry['outlet_length'].items():
                    l_val = float(len_str)
                    # Apply the natural log transformation as per original logic
                    points.append([dr, np.log(l_val + 1)])
                    values.append(c_val)
        
        if not points:
            raise ValueError(f"No data found for R/D={rd}")

        # 3. Prepare the target point for interpolation
        target_point = np.array([[diameter_ratio, np.log(outlet_length + 1)]])
        
        # 4. Perform 2D Interpolation
        c_value = griddata(np.array(points), np.array(values), target_point, method='linear')
        
        # Handle NaN if the target point is outside the convex hull of the data
        if np.isnan(c_value):
            # Fallback: nearest neighbor if outside the bounds
            c_value = griddata(np.array(points), np.array(values), target_point, method='nearest')

        return float(c_value[0])
        
    def _miller_outlet_correction_factor(self, rd:int, outlet_length:float, input_filename="Data Sources/miller_outlet_correction.csv"):
        # 0. Find the k_star
        k_star = self._ito_k(rd, self.flow.reynolds)/self._miller_interpolated_reynolds_correction(self.flow.reynolds)
        
        # 1. Load data
        df = pd.read_csv(input_filename)
        k_vals = df['K_star'].values
        len_vals = df['Outlet length'].values
        c_vals = df['C'].values
        
        # 2. Natural log transformation with +1 offset
        ln_len_orig = np.log1p(len_vals) 
        ln_len_input = np.log1p(outlet_length)
        
        # 3. Define the coordinate space (K*, ln(L+1))
        points = np.column_stack((k_vals, ln_len_orig))
        xi = np.array([[k_star, ln_len_input]])
        
        # 4. Interpolate
        c_value = griddata(points, c_vals, xi, method='linear')[0]
        if np.isnan(c_value):
            c_value = griddata(points, c_vals, xi, method='nearest')[0]
            
        return float(c_value)
        
    # Finding scramble coefficient
    
    def _miller_scramble_coefficient(self, rd1:float, rd2:float, seperation:float, angles=[0,30,60,90,120,150,180])->tuple[float,float,float]:
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
        
        for angle in angles:
            # Get correction factor for this specific orientation
            correction = self._miller_interpolated_elbow_correction(rd1, rd2, seperation, angle)
            
            # Calculate individual scramble
            k1_star = self._ito_k(rd1, self.flow.reynolds) / self._miller_interpolated_reynolds_correction(self.flow.reynolds)
            k2_star = self._ito_k(rd2, self.flow.reynolds) / self._miller_interpolated_reynolds_correction(self.flow.reynolds)
            s = (correction*(k1_star+k2_star) - k1_star*self._miller_outlet_correction_factor(rd1,seperation))/k2_star
            angle_scrambles.append(s)
        
        # Calculate statistics
        avg_scramble = sum(angle_scrambles) / len(angle_scrambles)
        min_scramble = min(angle_scrambles)
        max_scramble = max(angle_scrambles)
        
        return avg_scramble, min_scramble, max_scramble
    
    def _turner_scramble_coefficient(self, rd1:float, rd2:float, seperation:float, angles=[0,180], diameter_ratio=1)->tuple[float, float, float]:
        angle_scrambles = []

        for angle in angles:
            # Get correction factor for this specific orientation
            correction = self._miller_interpolated_elbow_correction(rd1, rd2, seperation, angle)
            
            # Calculate individual scramble
            k1 = self._turner_elbow_k(rd1, self.flow.reynolds, diameter_ratio)
            k2= self._turner_elbow_k(rd1, self.flow.reynolds, diameter_ratio)
            s = (correction*(k1+k2) - k1*self._turner_outlet_correction_factor(rd1,diameter_ratio,seperation))/k2
            angle_scrambles.append(s)
        
        # Calculate statistics
        avg_scramble = sum(angle_scrambles) / len(angle_scrambles)
        min_scramble = min(angle_scrambles)
        max_scramble = max(angle_scrambles)
        
        return avg_scramble, min_scramble, max_scramble
    
    # Find the K value
    def _miller_scramble_k(self, k:float, scramble_coefficient:float, re:float, curvature:float, outflow_length:float)->float:
        """Finds the K using the scramble and the outlet correction

        Args:
            k (float): Pressure loss coefficent [-]
            re (float): Reynolds number [-]
            curvature (float): Radius of curvature/Diameter [-]
            inflow_length (float): Inflow length/Diameter [-]
            outflow_length (float): Outflow length/Diameter [-]

        Returns:
            float: k
        """
        k_star = k/miller.interpolate_re_correction(re)
        outlet_correction = miller.interpolate_k_outlet_correction(k_star,outflow_length)
        
        return k*outlet_correction*scramble_coefficient





#METHODS   



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


# Uncluster this mess so I can choose to use Miller or turner methods, 


flow = Flow(5,998,1E-3,10E-3)    
data = Data("ito", flow)
print(data.get_outlet_correction_factor(2,5,1.1))
