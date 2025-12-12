import numpy as np
import os
#os.environ["PATH"] = "/storage/work/vai5027/.conda/envs/openmc-env1/bin:" + os.environ["PATH"]
import openmc
# --- UPDATED CROSS-SECTION PATH ---
#os.environ["OPENMC_CROSS_SECTIONS"] = "/storage/work/ajg7072/NUCE_403/endf/cross_sections.xml"
# The OpenMC configuration variable uses the standard key:
XS_PATH = "/storage/work/ajg7072/NUCE_403/endf/cross_sections.xml"
openmc.config['cross_sections'] = XS_PATH

# ----------------------------------

import math
import matplotlib.pyplot as plt
from openmc.model import borated_water
import pandas as pd
import shutil # For cleanup

# ----------------------------------------------------------------------
# 1. Defining materials
# ----------------------------------------------------------------------

T_ref = 294 # K

# Helper to ensure all materials are defined before geometry
def create_materials():
    fuel = openmc.Material(name='uo2')
    fuel.add_nuclide('U235', 0.04)
    fuel.add_nuclide('U238', 0.96)
    fuel.add_nuclide('O16', 2)
    fuel.set_density('g/cm3', 11.5)
    fuel.temperature = T_ref

    cladding = openmc.Material(name='zircaloy4')
    cladding.add_element('Zr',0.98)
    cladding.add_element('Sn',0.015)
    cladding.add_element('Fe',0.002)
    cladding.add_element('Cr',0.001)
    cladding.add_element('O',0.001)
    cladding.add_element('Hf', 0.001)
    cladding.set_density('g/cm3', 6.34)
    cladding.temperature = T_ref

    water = openmc.Material(name='h2o_unborated')
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 1.0)
    water.add_s_alpha_beta('c_H_in_H2O')
    water.temperature = T_ref

    boron_ppm_concentration = 1000
    temperature_k = 600
    pressure_mpa = 15 # MPa

    # This is the moderator material used in the core
    water_material = openmc.model.borated_water(
        boron_ppm=boron_ppm_concentration,
        temperature=temperature_k,
        pressure=pressure_mpa,
        name='borated_water_mod'
    )
    water_material.add_s_alpha_beta('c_H_in_H2O')

    ctrl_rod = openmc.Material(name='bc4')
    ctrl_rod.add_nuclide('B10', 2.8)
    ctrl_rod.add_nuclide('B11', 1.2)
    ctrl_rod.add_nuclide('C12', 1)
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

    mats = openmc.Materials([fuel, cladding, water, water_material, ctrl_rod, rpv])
    mats.export_to_xml()
    
    return mats, fuel, cladding, water_material, ctrl_rod, rpv

mats, fuel, cladding, water_material, ctrl_rod, rpv = create_materials()

# ----------------------------------------------------------------------
# 2. Geometry Dimensions and Surfaces
# ----------------------------------------------------------------------

H_core = 225
R_core = 70
cyl_uo2 = openmc.ZCylinder(r=0.6)
cyl_clad = openmc.ZCylinder(r=0.62)
cyl_ctrl = openmc.ZCylinder(r=0.62)
cyl_rpv_i = openmc.ZCylinder(r=R_core-5)
cyl_rpv_o = openmc.ZCylinder(r=R_core, boundary_type='vacuum')
z_max = openmc.ZPlane(z0=H_core, boundary_type='vacuum')
z_min = openmc.ZPlane(z0=-H_core, boundary_type='vacuum')

# Regions
uo2_region = -cyl_uo2 & -z_max & +z_min
clad_region = +cyl_uo2 & -cyl_clad & -z_max & +z_min
water_region = +cyl_clad & -z_max & +z_min # Moderator outside fuel pin
ctrl_region = -cyl_ctrl & -z_max & +z_min # Inner region of the control/guide tube
water_ctrl_region = +cyl_ctrl & -z_max & +z_min # Outer region of the control/guide tube

rpv_region = +cyl_rpv_i & -cyl_rpv_o & -z_max & +z_min

# --- Base Cells (shared by all states) ---
uo2_cell = openmc.Cell(name='fuel', region=uo2_region, fill=fuel)
clad_cell = openmc.Cell(name='cladding', region=clad_region, fill=cladding)
water_cell = openmc.Cell(name='water_material', region=water_region, fill=water_material)
rpv_cell = openmc.Cell(name='vessel', region=rpv_region, fill=rpv)

# --- Universes (for lattice components) ---
fuel_universe = openmc.Universe(cells=[uo2_cell, clad_cell, water_cell])
outer_universe = openmc.Universe(cells=[openmc.Cell(fill=water_material)])

# --- Control Rod Universes (Define ACI and ACO states for the center pin) ---

# ACI state (Control IN): Central rod filled with absorber (B4C)
ctrl_cell_aci = openmc.Cell(name='ctrl_rod_aci', region=ctrl_region, fill=ctrl_rod)
water_cell_ctrl_aci = openmc.Cell(name='water_ctrl_aci', region=water_ctrl_region, fill=water_material)
ctrl_in_universe = openmc.Universe(cells=[ctrl_cell_aci, water_cell_ctrl_aci])

# ACO state (Control OUT): Central rod filled with moderator (Borated water)
ctrl_cell_aco = openmc.Cell(name='ctrl_rod_aco', region=ctrl_region, fill=water_material)
water_cell_ctrl_aco = openmc.Cell(name='water_ctrl_aco', region=water_ctrl_region, fill=water_material)
ctrl_out_universe = openmc.Universe(cells=[ctrl_cell_aco, water_cell_ctrl_aco])


def create_full_model(central_pin_univ, model_name):
    """Creates a full OpenMC model for ACI or ACO configuration."""

    # --- Pin Lattice (Assembly) ---
    lat = openmc.HexLattice()
    lat.center = (0.0, 0.0)
    lat.pitch = (2.35,)
    lat.outer = outer_universe

    outer_ring = [fuel_universe] * 12
    middle_ring = [fuel_universe] * 6
    inner_ring = [central_pin_univ] # Central pin is the control/guide tube
    lat.universes = [outer_ring, middle_ring, inner_ring]

    a = 2.75 * lat.pitch[0]
    outer_boundary = openmc.model.hexagonal_prism(edge_length=a, orientation='y')
    main_cell = openmc.Cell(fill=lat, region=outer_boundary & -z_max & +z_min)
    assembly_univ = openmc.Universe(cells=[main_cell])

    # --- Core Lattice ---
    core_lat = openmc.HexLattice()
    core_lat.center = (0.,0.)
    core_lat.pitch = (np.sqrt(3)*a,)
    core_lat.outer = outer_universe
    core_lat.orientation = 'x'

    ring_1 = [assembly_univ]
    ring_2 = [assembly_univ] * 6
    ring_3 = [assembly_univ] * 12
    ring_4 = [assembly_univ] * 18
    ring_5 = [assembly_univ] * 24
    ring_6 = [assembly_univ] * 30
    core_lat.universes = [ring_6, ring_5, ring_4, ring_3, ring_2, ring_1]

    # Whole core cell for Tally target
    whole_core_cell_copy = openmc.Cell(
        name=f'{model_name}_core', 
        fill=core_lat, 
        region=-cyl_rpv_i & -z_max & +z_min
    )
    geom = openmc.Geometry([whole_core_cell_copy, rpv_cell])

    # --- Settings ---
    settings = openmc.Settings()
    settings.batches = 200
    settings.inactive = 30
    settings.particles = 50000
    settings.run_mode = 'eigenvalue'
    
    # Source distribution: box matching the core region
    core_radius = R_core - 5 # Same as inner vessel radius (cyl_rpv_i)
    source_space = openmc.stats.Box((-core_radius, -core_radius, -H_core),
                                    ( core_radius,  core_radius,  H_core))
    settings.source = openmc.IndependentSource(space=source_space)

    model = openmc.Model(geometry=geom, materials=mats, settings=settings)
    
    # Return the model and the specific cell object used for the core (tally target)
    return model, whole_core_cell_copy

# ----------------------------------------------------------------------
# 3. Tally Definition and Post-Processing
# ----------------------------------------------------------------------
def define_tallies(target_cell: openmc.Cell):
    """Defines tallies for flux spectrum and fission production (for normalization)."""
    
    # 1. Flux Spectrum Tally
    E_min = 1e-5
    E_max = 20e6
    num_bins = 500
    energies = np.logspace(np.log10(E_min), np.log10(E_max), num_bins + 1)
    e_filter = openmc.EnergyFilter(energies)
    cell_filter = openmc.CellFilter(target_cell) 
    
    flux_tally = openmc.Tally(name='flux_spectrum')
    flux_tally.filters = [e_filter, cell_filter]
    flux_tally.scores = ['flux']
    
    # 2. Fission Production Tally (REQUIRED FOR NORMALIZATION)
    # The cell filter on the core ensures we integrate over the whole core volume.
    fiss_tally = openmc.Tally(name='fiss_prod')
    fiss_tally.filters = [cell_filter]
    fiss_tally.scores = ['nu-fission']
    
    tallies = openmc.Tallies()
    tallies.append(flux_tally)
    tallies.append(fiss_tally)
    return tallies


CORE_POWER_W = 20e6  # Assuming a 20 MWt reactor
E_FISS_J = 3.204e-11  # Energy per fission (J/fission)

def process_spectrum(sp_file: str, name: str):
    """Loads tally, calculates flux per unit lethargy, and applies power normalization."""
    try:
        with openmc.StatePoint(sp_file) as sp:
            # fetch tallies (will raise KeyError if not present)
            t_flux = sp.get_tally(name='flux_spectrum')
            t_fiss = sp.get_tally(name='fiss_prod')

            # --- Compute simulated fission production and its uncertainty ---
            # t_fiss.mean and t_fiss.std_dev are arrays; call .sum() to reduce them to scalars
            P_sim = float(t_fiss.mean.sum())               # <--
            #CALL .sum()
            var_sim = float((t_fiss.std_dev**2).sum())    # variance sum
            P_std = math.sqrt(var_sim)

            # check for zero simulated fission production
            if P_sim == 0.0:
                raise RuntimeError("Simulated fission production (P_sim) is zero — cannot normalize flux.")

            # Actual power → number of fissions per second
            P_actual = CORE_POWER_W / E_FISS_J
            N = P_actual / P_sim

            # --- Flux processing (pandas DataFrame from tally) ---
            df = t_flux.get_pandas_dataframe()

            # ensure the expected columns exist
            if not {'energy low [eV]', 'energy high [eV]', 'mean'}.issubset(df.columns):
                raise KeyError(f"Unexpected Tally dataframe columns: {df.columns}")

            flux_mean = df['mean'].values
            E_lower = df['energy low [eV]'].values
            E_upper = df['energy high [eV]'].values

            # delta lethargy and normalized flux per unit lethargy
            delta_u = np.log(E_upper / E_lower)
            flux_per_u_norm = (flux_mean / delta_u) * N

            # k-eff scalar (use nominal_value)
            k_eff_scalar = float(sp.keff.nominal_value)

            data_length = len(flux_per_u_norm)
            k_eff_array = np.full(data_length, k_eff_scalar, dtype=float)

            result_df = pd.DataFrame({
                'E_mid [eV]': np.sqrt(E_lower * E_upper),
                'flux_per_u_norm': flux_per_u_norm,
                'k_eff': k_eff_array
            })

            # Print concise, informative diagnostics
            print(f"\n--- {name} Results ---")
            print(f"k_eff: {k_eff_scalar:.6f}")
            print(f"Total Fission Production Score (nu-fission): {P_sim:.6e} ± {P_std:.2e}")
            print(f"Normalization Factor N (applied to flux): {N:.6e}")

            return result_df

    except Exception as e:
        print(f"Error processing {name} results: {e}")
        return pd.DataFrame()

    
# ----------------------------------------------------------------------
# 4. ACI (Controls In) and ACO (Controls Out) EXECUTION
# ----------------------------------------------------------------------

# Clean up previous runs
if os.path.exists('run_aci'):
    shutil.rmtree('run_aci')
if os.path.exists('run_aco'):
    shutil.rmtree('run_aco')

# --- 1. Create Models and Define Tallies ---

# ACI: All Controls In (Central pin fill is B4C absorber)
model_aci, cell_aci = create_full_model(ctrl_in_universe, 'ACI')
model_aci.tallies = define_tallies(cell_aci)

# ACO: All Controls Out (Central pin fill is moderator)
model_aco, cell_aco = create_full_model(ctrl_out_universe, 'ACO')
model_aco.tallies = define_tallies(cell_aco)

# --- 2. Run Simulations ---
print("\nStarting ACI (Controls In) Simulation...")
# model.run() returns a PosixPath object. Convert to str for concatenation/use.
sp_aci_file = model_aci.run(output=False, cwd='run_aci')

print("Starting ACO (Controls Out) Simulation...")
sp_aco_file = model_aco.run(output=False, cwd='run_aco')

######

# --- 3. Post-Process Results (Applying the string/path fix) ---

# FIX: Convert PosixPath to string using str()
df_aci = process_spectrum(str(sp_aci_file), 'ACI (Controls In)')
df_aco = process_spectrum(str(sp_aco_file), 'ACO (Controls Out)')


# --- 4. Plot Flux Spectrum Per Unit Lethargy ---

if not df_aci.empty and not df_aco.empty:
    plt.figure(figsize=(10, 6))
    
    # --- FIX 2: Correct column name for plotting (KeyError Fix) ---
    plt.loglog(df_aci['E_mid [eV]'], df_aci['flux_per_u_norm'] * df_aci['E_mid [eV]'], 
               label=f'ACI (Controls In, $k_{{eff}}$={df_aci["k_eff"].iloc[0]:.4f})', linewidth=2)
    plt.loglog(df_aco['E_mid [eV]'], df_aco['flux_per_u_norm'] * df_aco['E_mid [eV]'], 
               label=f'ACO (Controls Out, $k_{{eff}}$={df_aco["k_eff"].iloc[0]:.4f})', linewidth=2)

    plt.xlabel('Energy E [eV]')
    plt.ylabel(r'$E \, \Phi(u)$  (Flux per Unit Lethargy)   [n/cm$^{2}$\ s]')
    plt.title('Core-Average Normalized Neutron Flux Spectrum (ACI vs ACO)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('potc4.png')
    
print("\nScript execution finished. Check 'run_aci' and 'run_aco' directories for statepoint files.")
print(df_aci)
print(df_aco)