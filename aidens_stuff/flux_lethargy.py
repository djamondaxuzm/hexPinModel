
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
import matplotlib.pyplot as plt
import openmc
import pandas as pd # <-- Added for post-processing
import shutil # <-- Added for cleaning up run directories

# ----------------------------------------------------------------------
# Cross section library - UPDATED PATH
# ----------------------------------------------------------------------
XS_PATH = "/storage/work/ajg7072/NUCE_403/endf/cross_sections.xml"
openmc.config['cross_sections'] = XS_PATH

# ----------------------------------------------------------------------
# Helper functions: temperature-dependent densities
# ----------------------------------------------------------------------

# Try to use IAPWS-97 for water at 15 MPa; fallback to simple approximation
try:
    from iapws import IAPWS97
    HAVE_IAPWS = True
except ImportError:
    HAVE_IAPWS = False

def water_density_15MPa_g_cm3(T_K: float) -> float:
    """Liquid water density [g/cm^3] at 15 MPa as a function of temperature [K]."""
    if HAVE_IAPWS:
        w = IAPWS97(P=15.0, T=T_K)      # P in MPa, T in K
        return w.rho / 1000.0          # kg/m^3 -> g/cm^3
    else:
        # Approximate near 15 MPa:
        T1, rho1 = 573.15, 0.72
        T2, rho2 = 623.15, 0.65
        if T_K <= T1:
            return rho1
        elif T_K >= T2:
            return rho2
        else:
            return rho1 + (rho2 - rho1) * (T_K - T1) / (T2 - T1)

def fuel_density_g_cm3(T_K: float,
                       T_ref: float = 600.0,
                       rho_ref: float = 11.5) -> float:
    """Simple fuel volumetric thermal expansion model for UO2(+Gd)."""
    alpha_lin = 1.0e-5
    alpha_vol = 3.0 * alpha_lin
    return rho_ref / (1.0 + alpha_vol * (T_K - T_ref))

# ----------------------------------------------------------------------
# Reference temperature for initial material definition
# ----------------------------------------------------------------------
T_ref = 600.0 # K

# ----------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------
fuel = openmc.Material(name='uo2_gad')
fuel.add_nuclide('U235', 0.04)
fuel.add_nuclide('U238', 0.96)
fuel.add_nuclide('O16', 2.0)
fuel.set_density('g/cm3', fuel_density_g_cm3(T_ref))
fuel.temperature = T_ref

cladding = openmc.Material(name='zircaloy4')
cladding.add_element('Zr', 0.98)
cladding.add_element('Sn', 0.015)
cladding.add_element('Fe', 0.002)
cladding.add_element('Cr', 0.001)
cladding.add_element('O', 0.001)
cladding.add_element('Hf', 0.001)
cladding.set_density('g/cm3', 6.34)
cladding.temperature = T_ref

water = openmc.Material(name='h2o')
water.add_nuclide('H1', 2.0)
water.add_nuclide('O16', 1.0)
water.set_density('g/cm3', water_density_15MPa_g_cm3(T_ref))
water.add_s_alpha_beta('c_H_in_H2O')
water.temperature = T_ref

ctrl_rod = openmc.Material(name='bc4')
ctrl_rod.add_nuclide('B10', 0.7)
ctrl_rod.add_nuclide('B11', 0.3)
ctrl_rod.add_nuclide('C12', 4.0)
ctrl_rod.set_density('g/cm3', 2.5)
ctrl_rod.temperature = T_ref

rpv = openmc.Material(name='stainless_steel')
rpv.add_element('Fe', 0.7)
rpv.add_element('Cr', 0.2)
rpv.add_element('Ni', 0.08)
rpv.add_element('Mn', 0.01)
rpv.add_element('C', 0.01)
rpv.set_density('g/cm3', 7.6)
rpv.temperature = T_ref

mats = openmc.Materials([fuel, cladding, water, ctrl_rod, rpv])

# ----------------------------------------------------------------------
# Geometry (Re-organized to create two states later)
# ----------------------------------------------------------------------
H_core = 225.0
R_core = 70.0    # cm

cyl_uo2 = openmc.ZCylinder(r=0.6)
cyl_clad = openmc.ZCylinder(r=0.62)
cyl_ctrl = openmc.ZCylinder(r=0.62)
cyl_rpv_i = openmc.ZCylinder(r=R_core - 10.0)
cyl_rpv_o = openmc.ZCylinder(r=R_core, boundary_type='vacuum')
z_max = openmc.ZPlane(z0=H_core,  boundary_type='vacuum')
z_min = openmc.ZPlane(z0=-H_core, boundary_type='vacuum')

# --- Shared Cells and Universes ---
uo2_cell = openmc.Cell(name='fuel', region=-cyl_uo2 & -z_max & +z_min, fill=fuel)
clad_cell = openmc.Cell(name='cladding', region=+cyl_uo2 & -cyl_clad & -z_max & +z_min, fill=cladding)
water_cell = openmc.Cell(name='water', region=+cyl_clad & -z_max & +z_min, fill=water) # Moderator outside pin
rpv_cell = openmc.Cell(name='vessel', region=+cyl_rpv_i & -cyl_rpv_o & -z_max & +z_min, fill=rpv)
outer_universe = openmc.Universe(cells=[openmc.Cell(fill=water)])
fuel_universe = openmc.Universe(cells=[uo2_cell, clad_cell, water_cell])
whole_core_cell = openmc.Cell(name='whole_core', region=-cyl_rpv_i & -z_max & +z_min) # Tally target

# --- Control Rod Universes ---
# 1. ACI (All Controls In): Central cell is the absorber
ctrl_rod_cell_in = openmc.Cell(name='ctrl_rod_in', region=-cyl_ctrl & -z_max & +z_min, fill=ctrl_rod)
water_cell_ctrl_in = openmc.Cell(name='water_ctrl_in', region=+cyl_ctrl & -z_max & +z_min, fill=water)
ctrl_in_universe = openmc.Universe(cells=[ctrl_rod_cell_in, water_cell_ctrl_in])

# 2. ACO (All Controls Out): Central cell is water
ctrl_rod_cell_out = openmc.Cell(name='ctrl_rod_out', region=-cyl_ctrl & -z_max & +z_min, fill=water) # Filled with water
water_cell_ctrl_out = openmc.Cell(name='water_ctrl_out', region=+cyl_ctrl & -z_max & +z_min, fill=water)
ctrl_out_universe = openmc.Universe(cells=[ctrl_rod_cell_out, water_cell_ctrl_out])


def create_core_model(ctrl_univ, model_name):
    """Creates a full OpenMC model for a given control rod state."""
    # --- Assembly Lattice (lat) ---
    lat = openmc.HexLattice()
    lat.center = (0.0, 0.0)
    lat.pitch  = (2.5,)
    lat.outer  = outer_universe
    
    # **Inner ring is the control rod universe**
    outer_ring  = [fuel_universe] * 12
    middle_ring = [fuel_universe] * 6
    inner_ring  = [ctrl_univ] 
    lat.universes = [outer_ring, middle_ring, inner_ring]
    
    a = 2.75 * lat.pitch[0]
    outer_boundary = openmc.model.hexagonal_prism(edge_length=a, orientation='y')
    main_cell = openmc.Cell(fill=lat, region=outer_boundary & -z_max & +z_min)
    assembly_univ = openmc.Universe(cells=[main_cell])
    
    # --- Core Lattice (core_lat) ---
    core_lat = openmc.HexLattice()
    core_lat.center = (0.0, 0.0)
    core_lat.pitch  = (np.sqrt(3.0) * a,)
    core_lat.outer  = outer_universe
    core_lat.orientation = 'x'

    ring_1 = [assembly_univ]
    ring_2 = [assembly_univ] * 6
    ring_3 = [assembly_univ] * 12
    ring_4 = [assembly_univ] * 18
    ring_5 = [assembly_univ] * 24
    core_lat.universes = [ring_5, ring_4, ring_3, ring_2, ring_1]

    # --- Geometry ---
    # Create a copy of the cell to ensure it is unique to this model
    core_cell_copy = whole_core_cell.clone()
    core_cell_copy.fill = core_lat 
    geom = openmc.Geometry([core_cell_copy, rpv_cell])

    # --- Settings ---
    settings = openmc.Settings()
    settings.batches   = 100
    settings.inactive  = 50
    settings.particles = 100_000
    settings.run_mode  = 'eigenvalue'
    settings.temperature = {
        'method': 'interpolation',
        'default': T_ref,
        'range': (300.0, 1500.0)
    }
    core_radius = R_core - 10.0
    source_space = openmc.stats.Box((-core_radius, -core_radius, -H_core),
                                    ( core_radius,  core_radius,  H_core))
    settings.source = openmc.IndependentSource(space=source_space)

    # --- Model ---
    model = openmc.Model(geometry=geom, materials=mats, settings=settings)
    
    # Return the model and the specific cell object used for the tally
    return model, core_cell_copy

# ----------------------------------------------------------------------
# Helper function: Flux Tally Definition
# ----------------------------------------------------------------------

def define_flux_spectrum_tally(target_cell: openmc.Cell):
    """Defines the Tally for flux per unit lethargy (Core Average)."""
    
    # 500 logarithmically spaced bins from 1e-5 eV to 20 MeV
    E_min = 1e-5
    E_max = 20e6
    num_bins = 500
    energies = np.logspace(np.log10(E_min), np.log10(E_max), num_bins + 1)
    e_filter = openmc.EnergyFilter(energies)

    cell_filter = openmc.CellFilter(target_cell) 
    
    tally = openmc.Tally(name='flux_spectrum')
    tally.filters = [e_filter, cell_filter]
    tally.scores = ['flux']
    
    return openmc.Tallies([tally])

# ----------------------------------------------------------------------
# Helper function: Post-Processing
# ----------------------------------------------------------------------

def process_spectrum(sp_file: str, name: str):
    """Loads tally, calculates flux per unit lethargy, and returns DataFrame."""
    try:
        with openmc.StatePoint(sp_file) as sp:
            t = sp.get_tally(name='flux_spectrum')
            
            # 1. Extract Flux and Energy Bins
            df = t.get_pandas_dataframe()
            flux_mean = df['mean'].values
            
            # Get energy boundaries for each bin
            E_lower = df['energy low [eV]'].values
            E_upper = df['energy high [eV]'].values
            
            # 2. Calculate Lethargy Width (Delta u)
            # Δu_g = ln(E_upper / E_lower)
            delta_u = np.log(E_upper / E_lower)
            
            # 3. Calculate Flux per Unit Lethargy: Φ(u) = Φ / Δu
            flux_per_u = flux_mean / delta_u
            
            # 4. Create result DataFrame
            result_df = pd.DataFrame({
                'E_mid [eV]': np.sqrt(E_lower * E_upper), 
                'flux_per_u': flux_per_u,
                'k_eff': sp.k_combined.n
            })
            
            print(f"\n--- {name} Results ---")
            print(f"k_eff: {sp.k_combined}")
            
            return result_df
            
    except Exception as e:
        print(f"Error processing {name} results: {e}")
        return pd.DataFrame()


# ----------------------------------------------------------------------
# ********** ACI (Controls In) and ACO (Controls Out) EXECUTION **********
# ----------------------------------------------------------------------

# Clean up previous runs if they exist
if os.path.exists('run_aci'):
    shutil.rmtree('run_aci')
if os.path.exists('run_aco'):
    shutil.rmtree('run_aco')

# --- 1. Create Models and Define Tallies ---

# ACI: All Controls In (Central pin fill is ctrl_rod)
model_aci, cell_aci = create_core_model(ctrl_in_universe, 'ACI')
model_aci.tallies = define_flux_spectrum_tally(cell_aci)

# ACO: All Controls Out (Central pin fill is water)
model_aco, cell_aco = create_core_model(ctrl_out_universe, 'ACO')
model_aco.tallies = define_flux_spectrum_tally(cell_aco)

# --- 2. Run Simulations ---
print("\nStarting ACI (Controls In) Simulation...")
sp_aci_file = model_aci.run(output=False, cwd='run_aci')

print("Starting ACO (Controls Out) Simulation...")
sp_aco_file = model_aco.run(output=False, cwd='run_aco')


# --- 3. Post-Process Results ---

df_aci = process_spectrum(str(sp_aci_file), 'ACI (Controls In)')
df_aco = process_spectrum(str(sp_aco_file), 'ACO (Controls Out)')


# --- 4. Plot Flux Spectrum Per Unit Lethargy ---

if not df_aci.empty and not df_aco.empty:
    plt.figure(figsize=(10, 6))
    
    # Plotting E*Phi(u) vs E is standard for spectral comparison
    # Area under E*Phi(u) is proportional to the total flux, making regions comparable.
    plt.loglog(df_aci['E_mid [eV]'], df_aci['flux_per_u'] * df_aci['E_mid [eV]'], 
               label=f'ACI (Controls In, $k_{{eff}}$={df_aci["k_eff"].iloc[0]:.4f})', linewidth=2)
    plt.loglog(df_aco['E_mid [eV]'], df_aco['flux_per_u'] * df_aco['E_mid [eV]'], 
               label=f'ACO (Controls Out, $k_{{eff}}$={df_aco["k_eff"].iloc[0]:.4f})', linewidth=2)

    plt.xlabel('Energy E [eV]')
    plt.ylabel('E $\Phi$(u) (Flux per Unit Lethargy) [Arbitrary Units/Source Particle]')
    plt.title('Core-Average Neutron Flux Spectrum per Unit Lethargy (ACI vs ACO)')
    plt.text(1e-1, 1e-4, 'Thermal Region', horizontalalignment='center', color='blue')
    plt.text(1e2, 1e-3, '$1/E$ (Epithermal) Region', horizontalalignment='center', color='green')
    plt.text(1e6, 1e-2, 'Fast Region', horizontalalignment='center', color='red')
    
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.show()
    

# ----------------------------------------------------------------------
# Original temperature coefficient script continues here
# ----------------------------------------------------------------------

# To run the temperature coefficient analysis, you must explicitly
# re-create and run one of the base models (e.g., ACO state) or 
# define a separate model instance for those calculations, as the 
# previous runs used the 'model_aci' and 'model_aco' instances.