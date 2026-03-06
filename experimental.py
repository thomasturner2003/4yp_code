import math


def flow_rate_to_speed(diameter:float, flow_rate:float)->float:
    """Converts L/min to m/s

    Args:
        diameter (float): [m]
        flow_rate (float): [L/min]

    Returns:
        float: flow speed [m/s]
    """
    area = math.pi*((diameter/2)**2)
    flow_rate_SI = flow_rate/(1000*60) # seconds in min x L in m3
    return flow_rate_SI/area

def flow_rate_to_reynolds(flow_rate:float, viscosity:float, density:float=998, diameter:float=10E-3)->float:
    """Converts L/min to Re

    Args:
        flow_rate (float): [L/min]
        viscosity (float): dynamic
        density (float, optional): [kgm-3] Defaults to 998.
        diameter (float, optional): [m] Defaults to 10E-3.

    Returns:
        float: Re[-]
    """
    speed = flow_rate_to_speed(diameter, flow_rate)
    return (density*speed*diameter)/(viscosity)

def blasius_dp(flow_rate:float, viscosity:float, length:float, density:float=998, diameter:float=10E-3)->float:
    """Finds the predicted blasius Dp

    Args:
        flow_rate (float): L/min
        viscosity (float): dynamic
        length (float): [m]
        density (float, optional): [kgm-3]. Defaults to 998.
        diameter (float, optional): [m]. Defaults to 10E-3.

    Returns:
        float: Pressure drop [Pa]
    """
    re = flow_rate_to_reynolds(flow_rate, viscosity, density=density, diameter=diameter)
    speed = flow_rate_to_speed(diameter,flow_rate)
    eps = 0.0015E-3
    friction_factor = 0.25 / (math.log10(eps/(3.7*diameter) + 5.74/(re**0.9))**2)
    #friction_factor = 0.3164/(re**0.25)
    return friction_factor*(length/diameter)*(density*speed**2)/2


def dp_to_darcy_friction_factor(dp:float, flow_rate:float, viscosity:float, length:float, density:float=998, diameter:float=10E-3)->float:
    """Finds the darcy friction factor of a pipe

    Args:
        dp (float): [Pa]
        flow_rate (float): [L/min]
        viscosity (float): dynamic
        length (float): [m]
        density (int, optional): _description_. Defaults to 998.
        diameter (_type_, optional): _description_. Defaults to 10E-3.

    Returns:
        _type_: friciton factor[-]
    """
    speed = flow_rate_to_speed(diameter, flow_rate)
    return dp/((length/diameter)*(0.5*density*speed**2))


def calculate_pipe_roughness(dp, length, flow_rate, viscosity, diameter=10E-3, density=998):
    """
    Calculates the absolute pipe roughness (epsilon) using the Colebrook equation.
    
    Parameters:
    f  : float - Darcy friction factor (dimensionless)
    re : float - Reynolds number (dimensionless)
    d  : float - Internal diameter of the pipe (meters)
    
    Returns:
    epsilon : float - Absolute roughness (meters)
    """
    friction_factor = dp_to_darcy_friction_factor(dp, flow_rate, viscosity, length)
    reynolds = flow_rate_to_reynolds(flow_rate,viscosity)
    # Rearranged Colebrook equation to solve for epsilon
    term1 = 10**(-1 / (2 * math.sqrt(friction_factor)))
    term2 = 2.51 / (reynolds * math.sqrt(friction_factor))
    
    epsilon = 3.7 * diameter * (term1 - term2)
    
    return epsilon


if __name__ == "__main__":
    #dps = [6784, 5971, 5330, 3668, 2730]
    #flow_rates = [14.9, 9.83, 12.31, 12.88, 11.89, 11.38, 10.16]
    #dps = [12521, 6040, 8729, 9556, 8115, 7543, 6136]
    #ls = [1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03]
    #
    raw_rate = 14.03
    rate = raw_rate*(1.182 - 0.0112*raw_rate)
    measured = 8.438E3
    viscosity = 1.1E-3
    length = 580E-3
    diameter = 10E-3
    k = 0.13 + 0.25
    momentum = 0.5*998*flow_rate_to_speed(diameter,rate)**2
    dp_k = momentum*k
    print(f"Adjusted flow rate: {rate}")
    print(f"Pressure drop due to pressure sensor: {dp_k}")
    dp = blasius_dp(rate,viscosity,length)
    print(f"Pressure drop due to pipe: {(measured-dp_k)}")
    print(f"Predicted pressure drop: {dp}")
    print(f"Error: {100*((measured-dp_k)-dp)/(measured-dp_k)}%")
    print(f"Reynolds: {flow_rate_to_reynolds(rate,viscosity)}")
    print(f"Flow speed: {flow_rate_to_speed(diameter,rate)}")
    