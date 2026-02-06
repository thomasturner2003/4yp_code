import math

def richardson_gci(phi3, phi2, phi1, r, Fs=1.25):
    # phi3 = coarse, phi2 = medium, phi1 = fine
    p = math.log((phi3 - phi2) / (phi2 - phi1)) / math.log(r)
    phi_ext = phi1 + (phi1 - phi2) / (r**p - 1)
    GCI_fine = Fs * abs(phi2 - phi1) / (abs(phi1) * (r**p - 1))
    GCI_med = Fs * abs(phi3 - phi2) / (abs(phi2) * (r**p - 1))
    return p, phi_ext, GCI_fine * 100, GCI_med * 100  # return % GCI values

def first_layer_height_blasius(U, D, rho, mu, y_plus):
    """
    Compute the physical first-layer cell height for a target y+ in turbulent 
    pipe flow using the Blasius correlation for smooth pipes.

    Parameters
    ----------
    U : float
        Bulk flow velocity [m/s]
    D : float
        Pipe internal diameter [m]
    rho : float
        Fluid density [kg/m^3]
    mu : float
        Dynamic viscosity [Pa.s]
    y_plus : float
        Target non-dimensional wall distance y+ [-]

    Returns
    -------
    y : float
        First-layer physical cell height [m]
    """
    # 1. Kinematic Viscosity and Reynolds Number
    nu = mu / rho
    Re = (U * D) / nu 
    
    # 2. Blasius Friction Factor (Valid for smooth pipes, Re < 10^5)
    f = 0.316 * (Re**-0.25)
    
    # 3. Wall Shear Stress and Friction Velocity
    # tau_w = (f * rho * U^2) / 8
    # u_tau = sqrt(tau_w / rho) -> simplifies to:
    u_tau = U * (f / 8.0)**0.5
    
    # 4. Physical Height y
    # y = (y_plus * nu) / u_tau
    y = 2*(y_plus * nu) / u_tau
    
    return y

def first_layer_height(U, D, rho, mu, y_plus, eps=0.0):
    """
    Compute the physical first-layer cell height for a target y+ in turbulent pipe flow
    using the Colebrook equation for friction factor.

    Parameters
    ----------
    U : float
        Bulk flow velocity [m/s]
    D : float
        Pipe internal diameter [m]
    rho: float
        Density (kg/m^3)
    mu : float
        Dynamic viscosity [m^2/s]
    y_plus : float
        Target non-dimensional wall distance y+ [-]
    eps : float, optional
        Absolute pipe-wall roughness [m], default = 0 (smooth pipe)

    Returns
    -------
    y : float
        First-layer physical cell height [m]
    """
    nu = mu/rho
    Re = U * D/nu 
    epsD = eps / D
    f = 0.02
    for _ in range(50):
        rhs = -2.0 * math.log10(epsD / 3.7 + 2.51 / (Re * math.sqrt(f)))
        f = 1.0 / (rhs * rhs)
    return y_plus * nu / (U * math.sqrt(f / 8.0))

def total_height(h0, growth, N):
    return h0*(1-growth**N)/(1-growth)

def transition_ratio(h0, N, mesh, growth=1.2):
    return h0*growth**N/mesh

def first_thickness_and_layers(U, D, mu, rho, yplus_first, yplus_total, growth, eps=0):
    nu  = mu/rho
    Re  = U*D/nu
    f   = 0.02
    for _ in range(50):
        rhs = -2.0*math.log10(eps/D/3.7 + 2.51/(Re*math.sqrt(f)))
        f   = 1/(rhs*rhs)

    u_tau = U*math.sqrt(f/8.0)

    h0 = yplus_first * nu / u_tau          # first cell height
    H  = yplus_total * nu / u_tau          # total BL target height
    
    if abs(growth-1) < 1e-12:
        N = math.ceil(H/h0)
    else:
        N = math.ceil(math.log(1+(growth-1)*H/h0)/math.log(growth))
    H_bar = total_height(h0,growth,N)
    return h0, N, H, H_bar

def first_thickness_and_layers_trans(U, D, mu, rho, yplus_first, transition, mesh, growth, eps=0):
    nu  = mu/rho
    Re  = U*D/nu
    f   = 0.02
    for _ in range(50):
        rhs = -2.0*math.log10(eps/D/3.7 + 2.51/(Re*math.sqrt(f)))
        f   = 1/(rhs*rhs)

    u_tau = U*math.sqrt(f/8.0)

    h0 = 2*yplus_first * nu / u_tau          # first cell height
    N = 1
    h = h0
    H = h0
    while True:
        tr = h/mesh
        if tr > transition:
            break
        N+=1
        h *=growth
        H+= h
    return h0, N, H

def calculate_inflation_total_height(base_mesh_size, aspect_ratio, h1, num_layers):
    """
    Calculates total inflation height using the Last Aspect Ratio method.
    
    Logic:
    1. Define the target height of the final layer (h_last).
    2. Calculate the required growth rate (r) to reach h_last from h1.
    3. Sum the geometric series to find the total thickness.
    """
    
    # 1. Target height of the last inflation layer
    aspect_ratio = 1/aspect_ratio
    h_last = base_mesh_size * aspect_ratio
    
    # 2. Calculate Growth Rate (r)
    # Formula: h_last = h1 * r^(num_layers - 1)
    # Therefore: r = (h_last / h1) ^ (1 / (num_layers - 1))
    growth_rate = (h_last / h1) ** (1 / (num_layers - 1))
    
    # 3. Sum of Geometric Series
    # Total Height = h1 * (1 - r^n) / (1 - r)
    total_height = h1 * (1 - growth_rate**num_layers) / (1 - growth_rate)
    
    return total_height

def calculate_y_plus(y, v_bulk, D, rho, mu):
    """
    Calculates y+ at a specific height y using the Blasius correlation.
    
    Assumptions:
    1. Smooth pipe (relative roughness = 0).
    2. Re < 10^5 (valid range for Blasius).
    3. Fully developed turbulent flow.
    """
    
    # 1. Calculate Reynolds Number
    Re = (rho * v_bulk * D) / mu
    
    # 2. Blasius Friction Factor (f)
    # f = 0.316 * Re^(-0.25)
    f = 0.316 * (Re**-0.25)
    
    # 3. Calculate Wall Shear Stress (tau_w)
    # tau_w = (f * rho * v_bulk^2) / 8
    tau_w = (f * rho * v_bulk**2) / 8
    
    # 4. Calculate Friction Velocity (u_tau)
    u_tau = (tau_w / rho)**0.5
    
    # 5. Calculate y+
    # y+ = (y * u_tau * rho) / mu
    y_plus = (y * u_tau * rho) / mu
    
    return y_plus


print(2*first_layer_height(3, 0.01, 998, 1E-3, 1))
print(calculate_inflation_total_height(0.3E-3, 3, 7.8E-6, 25)*1000)