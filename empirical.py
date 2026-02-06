import math


def reynolds(rho, V, D, mu):
    """Reynolds number"""
    return rho * V * D / mu

def swamee_jain_f(Re, eps, D):
    """Swamee-Jain explicit approximation for turbulent flow friction factor.
       Valid for Re >= ~5000 (approx) and rough/turbulent ranges."""
    if Re <= 0:
        raise ValueError("Re must be > 0")
    return 0.25 / (math.log10(eps/(3.7*D) + 5.74/(Re**0.9))**2)

def blasius_darcy(L, rho=998, V=3, D=13e-3, mu=1e-3):
    re = reynolds(rho, V, D, mu)
    f = 0.3164*(re**-0.25)
    return f*(L/D)*(rho*V**2)/2

def colebrook_f_iterative(Re, eps, D, tol=1e-6, maxiter=50):
    """Solve Colebrook equation by fixed-point iteration (or Newton-style update).
       Returns Darcy friction factor f.
       Colebrook: 1/sqrt(f) = -2.0*log10( (eps/(3.7D)) + (2.51/(Re*sqrt(f))) )
    """
    if Re < 2300:
        return 64.0 / Re
    # initial guess: Swamee-Jain
    f = swamee_jain_f(Re, eps, D)
    for i in range(maxiter):
        # compute function value for current f
        lhs = 1.0 / math.sqrt(f)
        rhs = -2.0 * math.log10(eps/(3.7*D) + 2.51/(Re * math.sqrt(f)))
        # simple relaxation update (Aitken-style could be used)
        f_new = (1.0 / rhs**2)
        if abs(f_new - f) < tol:
            return f_new
        f = f_new
    # warn but return last
    return f

def pressure_drop_darcy(rho, V, D, L, mu, eps=1.5e-5, K_minor=0.0, use_colebrook=False):
    """
    Compute pressure drop (Pa) for incompressible flow using Darcy-Weisbach.
    - rho: density (kg/m3)
    - V: mean velocity (m/s)
    - D: diameter (m)
    - L: pipe length (m)
    - mu: dynamic viscosity (Pa.s)
    - eps: absolute roughness (m), default ~ smooth commercial steel ~1.5e-5 m
    - K_minor: sum of minor loss coefficients (dimensionless)
    - use_colebrook: if True solve Colebrook iteratively; otherwise use Swamee-Jain
    """
    g = 9.80665
    Re = reynolds(rho, V, D, mu)
    if Re < 2300:
        f = 64.0 / Re
    else:
        if use_colebrook:
            f = colebrook_f_iterative(Re, eps, D)
        else:
            f = swamee_jain_f(Re, eps, D)
    # head loss
    hf = f * (L / D) * (V**2) / (2*g)      # meters of fluid
    dp_pipe = rho * g * hf                # Pa
    dp_minor = K_minor * 0.5 * rho * V**2 # Pa
    dp_total = dp_pipe + dp_minor
    return {
        "Re": Re,
        "f": f,
        "hf_m": hf,
        "dp_pipe_Pa": dp_pipe,
        "dp_minor_Pa": dp_minor,
        "dp_total_Pa": dp_total
    }
    
def predicted_K(rho, V, dp):
    return (dp)/(0.5*rho*V*V)

    try:
        case = data.read_case(case_ID)
        fluid = data.read_fluid(case['Working fluid'])
        disturbance_1 = data.read_disturbance(case['Disturbance 1'])
        disturbance_2 = data.read_disturbance(case['Disturbance 2'])
        disturbance_3 = data.read_disturbance(case['Disturbance 3'])
        disturbance_4 = data.read_disturbance(case['Disturbance 4'])
        disturbance_5 = data.read_disturbance(case['Disturbance 5'])
        disturbances= [disturbance_1, disturbance_2, disturbance_3, disturbance_4, disturbance_5]
        k = 0;
        for disturbance in disturbances:
            k += float(disturbance['K'])
        res = pressure_drop_darcy(float(fluid['Rho']), float(case['Velocity']), float(case['Diameter']), float(case['Length']), float(fluid['Mu']), float(case['Roughness']), k, use_colebrook=True)
        data.write_result(case_ID, "Empirical", res["dp_total_Pa"])
        return res
    except:
        print("⚠️  Warning - Test failed!")
        
def find_U_length(inlet_outlet_length, seperation, bend_radius):
    return (2*inlet_outlet_length + seperation + 2*(2*bend_radius*math.pi/4))/1000
