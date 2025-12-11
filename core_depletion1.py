import numpy as np
import os
import math
import openmc
from openmc.model import borated_water
os.environ["PATH"] = "/storage/work/vai5027/.conda/envs/openmc-env1/bin:" + os.environ["PATH"]

os.environ["OPENMC_CROSS_SECTIONS"] = "/storage/home/vai5027/work/NUCE403/endfb-viii/cross_sections.xml"
## Defining materials


fuel = openmc.Material(name='uo2')
fuel.add_nuclide('U235', 0.04)
fuel.add_nuclide('U238', 0.96)
fuel.add_nuclide('O16', 2)
fuel.set_density('g/cm3', 11.5)

cladding = openmc.Material(name='zircaloy4')
cladding.add_element('Zr',0.98)
cladding.add_element('Sn',0.015)
cladding.add_element('Fe',0.002)
cladding.add_element('Cr',0.001)
cladding.add_element('O',0.001)
cladding.add_element('Hf', 0.001)
cladding.set_density('g/cm3', 6.34)

water = openmc.Material(name='h2o')
water.add_nuclide('H1', 2.0)
water.add_nuclide('O16', 1.0)
water.set_density('g/cm3', 1.0)
water.add_s_alpha_beta('c_H_in_H2O')

boron_ppm_concentration = 1000
temperature_k = 600.0
pressure_mpa = 15 # MPa

water_material = openmc.model.borated_water(
    boron_ppm=boron_ppm_concentration,
    temperature=temperature_k,
    pressure=pressure_mpa
)

ctrl_rod = openmc.Material(name='bc4')
ctrl_rod.add_nuclide('B10', 0.7)
ctrl_rod.add_nuclide('B11', 0.3)
ctrl_rod.add_nuclide('C12', 4)
ctrl_rod.set_density('g/cm3', 2.5)

rpv = openmc.Material(name='stainless_steel')   #Not really following any stablished "named" materials, just went with an approximation
rpv.add_element('Fe', 0.7)
rpv.add_element('Cr', 0.2)
rpv.add_element('Ni', 0.08)
rpv.add_element('Mn', 0.01)
rpv.add_element('C', 0.01)
rpv.set_density('g/cm3', 7.6)

mats = openmc.Materials([fuel, cladding, water_material, ctrl_rod, rpv])
mats.export_to_xml()



## Defining geometry


# First defining dimensions and limiting surfaces for fuel, clad, and (when we had them) control rods
# Specify boundary conditions only to the exterior surfaces
# Changed the radius of the pellet to be more realistic (1 whole cm was a lot)

H_core = 225
R_core = 70
cyl_uo2 = openmc.ZCylinder(r=0.6)
cyl_clad = openmc.ZCylinder(r=0.62)
cyl_ctrl = openmc.ZCylinder(r=0.62)
cyl_no_ctrl = openmc.ZCylinder(r=0.62)
cyl_rpv_i = openmc.ZCylinder(r=R_core-5)
cyl_rpv_o = openmc.ZCylinder(r=R_core, boundary_type='vacuum')
z_max = openmc.ZPlane(z0=H_core, boundary_type='vacuum')
z_min = openmc.ZPlane(z0=-H_core, boundary_type='vacuum')

# regions
uo2_region = -cyl_uo2 & -z_max & +z_min
clad_region = +cyl_uo2 & -cyl_clad & -z_max & +z_min
water_region = +cyl_clad & -z_max & +z_min
ctrl_region = -cyl_ctrl & -z_max & +z_min
no_ctrl_region = -cyl_no_ctrl & -z_max & +z_min
water_ctrl_region = +cyl_ctrl & -z_max & +z_min
water_no_ctrl_region = +cyl_no_ctrl & -z_max & +z_min
rpv_region = +cyl_rpv_i & -cyl_rpv_o & -z_max & +z_min
water_rod_region = -cyl_ctrl & -z_max & +z_min

# cells
uo2_cell = openmc.Cell(name='fuel')
uo2_cell.region = uo2_region
uo2_cell.fill = fuel

clad_cell = openmc.Cell(name='cladding')
clad_cell.region = clad_region
clad_cell.fill = cladding

ctrl_cell = openmc.Cell(name='ctrl_rod')
ctrl_cell.region = ctrl_region
ctrl_cell.fill = ctrl_rod

no_ctrl_cell = openmc.Cell(name='no_ctrl_rod')
no_ctrl_cell.region = no_ctrl_region
no_ctrl_cell.fill = water_material

water_cell = openmc.Cell(name='water_material')
water_cell.region = water_region
water_cell.fill = water_material

water_cell_ctrl = openmc.Cell(name='water_ctrl')        # This might be unnecessary, essentially the same as the water cell
water_cell_ctrl.region = water_ctrl_region
water_cell_ctrl.fill = water_material

water_cell_no_ctrl = openmc.Cell(name='water_no_ctrl')        # This might be unnecessary, essentially the same as the water cell
water_cell_no_ctrl.region = water_no_ctrl_region
water_cell_no_ctrl.fill = water_material

rpv_cell = openmc.Cell(name='vessel')
rpv_cell.region = rpv_region
rpv_cell.fill = rpv

water_rod_cell = openmc.Cell(name='water_rod_uncontrolled')
water_rod_cell.region = water_rod_region
water_rod_cell.fill = water_material

# universes, the outer universe i see it as a safety net so that everything is covered

fuel_universe = openmc.Universe(cells=[uo2_cell, clad_cell, water_cell])
ctrl_rod_universe = openmc.Universe(cells=[ctrl_cell, water_cell_ctrl])
outer_universe = openmc.Universe(cells=[openmc.Cell(fill=water_material)])
no_ctrl_rod_universe = openmc.Universe(cells=[no_ctrl_cell, water_cell_no_ctrl])

lat = openmc.HexLattice()    # this is the lattice of the fuel pins arranged into 3 rings to form the assembly
lat.center = (0.0, 0.0)
lat.pitch = (2.5,)           # i picked this number just to make it fit, no calculations.
# Changing this number affects k-eff a lot. Could make it smaller and even fit one more assembly ring
lat.outer = outer_universe

outer_ring = [fuel_universe] * 12
middle_ring = [fuel_universe] * 6
inner_ring = [no_ctrl_rod_universe]         # this used to be ctrl_rod_universe, but since we need burnable poisons it needs to be uncontrolled
lat.universes = [outer_ring, middle_ring, inner_ring] # this completely defines the lattice structure


# this defines the fuel assembly cell so that it can be stacked
a = 2.75 * lat.pitch[0]  # formula
outer_boundary = openmc.model.hexagonal_prism(edge_length=a, orientation='y')
main_cell = openmc.Cell(fill=lat, region=outer_boundary & -z_max & +z_min)

assembly_univ = openmc.Universe(cells=[main_cell]) #just "converting" the cell into a universe so that it can be merged

core_lat = openmc.HexLattice()
core_lat.center = (0.,0.)
core_lat.pitch = (np.sqrt(3)*a,)  # the sqrt3*a is the formula, then manually adjusting it so that it looks homogeneous
core_lat.outer = outer_universe # used the same outer as when defining the assemblies, it does not matter
core_lat.orientation = 'x' # these orientations are either x or y. I just change them until they agree lol

ring_1 = [assembly_univ]
ring_2 = [assembly_univ] * 6
ring_3 = [assembly_univ] * 12
ring_4 = [assembly_univ] * 18
ring_5 = [assembly_univ] * 24
ring_6 = [assembly_univ] * 30
# ring_7 = [assembly_univ] * 36
core_lat.universes = [ring_6, ring_5, ring_4, ring_3, ring_2, ring_1]

# only the "water" of the core, no pressure vessel
whole_core_cell = openmc.Cell(fill=core_lat, region=-cyl_rpv_i & -z_max & +z_min)

geom = openmc.Geometry([whole_core_cell, rpv_cell]) #pressure vessel added here
geom.export_to_xml()





settings = openmc.Settings()
settings.batches = 300
settings.inactive = 50
settings.particles = 100000
settings.run_mode = 'eigenvalue'
settings.export_to_xml()





# Depletion Cell
import openmc.deplete
fuel.volume = np.pi * (cyl_uo2.r**2) * H_core * 1638
chain = openmc.deplete.Chain.from_xml("chain_endfb71_pwr.xml")
model = openmc.Model(geometry=geom, settings=settings)
operator = openmc.deplete.CoupledOperator(model, chain)
power = 20e6 # W
time_steps = [0.1,0.2,0.4,0.8,1.6,2.5,5,10,20,40,80,100,200,400,800]
cecm = openmc.deplete.CECMIntegrator(operator, time_steps, power, timestep_units='d') # also better options available
cecm.integrate()






# Depletion Post Processing

from matplotlib import pyplot as plt

results = openmc.deplete.Results("./depletion_results.h5")
time, k = results.get_keff()
time /= (24 * 60 * 60)  # convert seconds → days

keff_mean = k[:, 0]
keff_sigma = k[:, 1]

plt.errorbar(time, keff_mean, yerr=keff_sigma, fmt='o-', capsize=3)
plt.xlabel("Time [d]")
plt.ylabel(r"$k_{\mathrm{eff}} \pm \sigma$")
plt.tight_layout()
plt.savefig("HIVE_Depletion.png", dpi=300)

