import math

def pipe_length(flow_rate_lpm, pressure_drop, diameter_mm):
    """
    Calculate pipe length for given flow rate, pressure drop, and pipe diameter.
    Assumes water at ~20°C and smooth copper pipe.

    flow_rate_lpm: Flow rate in litres per minute
    pressure_drop: Total pressure drop in kPa
    diameter_mm: Internal diameter of pipe in mm
    """
    # Constants
    rho = 998          # kg/m³, water density
    mu = 1.002e-3      # Pa·s, dynamic viscosity
    g = 9.81           # m/s²

    # Unit conversions
    Q = flow_rate_lpm / 1000 / 60  # m³/s
    D = diameter_mm / 1000         # m
    dp = pressure_drop * 1000

    # Flow velocity
    A = math.pi * (D / 2) ** 2
    v = Q / A

    # Reynolds number
    Re = rho * v * D / mu

    # Relative roughness for copper (very smooth)
    epsilon = 0.0015e-3
    rel_roughness = epsilon / D

    # Swamee-Jain equation for friction factor (valid for Re > 4000)
    f = 0.25 / (math.log10((rel_roughness / 3.7) + (5.74 / (Re ** 0.9)))) ** 2

    # Darcy–Weisbach: dp = f*(L/D)*(rho*v²/2)
    # Rearranged for L:
    L = (dp * D) / (f * 0.5 * rho * v ** 2)

    return {
        "Flow velocity (m/s)": round(v, 3),
        "Reynolds number": round(Re, 0),
        "Friction factor": round(f, 4),
        "Pipe length (m)": round(L, 2)
    }


def orifice_diameter_mm(flow_lpm, dp_kpa, pipe_d_mm, Cd=0.61, rho=998.0):
    """
    Calculate the orifice inner diameter (mm) that gives the desired flow
    for a given pressure drop (kPa) and pipe diameter (mm).

    Uses the incompressible flow orifice equation:
        Q = Cd * A * sqrt(2 * dp / rho)

    Parameters
    ----------
    flow_lpm : float   # flow rate in L/min
    dp_kpa : float     # pressure drop across orifice in kPa
    pipe_d_mm : float  # internal diameter of pipe in mm
    Cd : float         # discharge coefficient (default 0.61)
    rho : float        # fluid density (kg/m³), default water ~998

    Returns
    -------
    dict with:
        - orifice_d_mm: calculated orifice diameter
        - beta_ratio: d_orifice / d_pipe
        - velocity_m_s: velocity through orifice
        - Re: Reynolds number based on orifice diameter
    """
    Q = flow_lpm / 1000 / 60     # m³/s
    dp = dp_kpa * 1000           # Pa
    D = pipe_d_mm / 1000         # m

    # Orifice area and diameter
    A = Q / (Cd * math.sqrt(2 * dp / rho))
    d = math.sqrt(4 * A / math.pi)

    beta = d / D
    v = Q / A
    Re = rho * v * d / 1.002e-3  # approximate for water @ 20°C

    return {
        "orifice_d_mm": d * 1000,
        "pipe_d_mm": pipe_d_mm,
        "beta_ratio": beta,
        "velocity_m_s": v,
        "Re": Re,
        "Cd": Cd
    }


# Example usage
result = orifice_diameter_mm(flow_lpm=2, dp_kpa=82, pipe_d_mm=13.6)
for k, v in result.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4g}")
    else:
        print(f"{k}: {v}")



