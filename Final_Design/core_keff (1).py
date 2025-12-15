import numpy as np
import os

os.environ["PATH"] = "/storage/work/vai5027/.conda/envs/openmc-env1/bin:" + os.environ["PATH"]
os.environ["OPENMC_CROSS_SECTIONS"] = "/storage/home/vai5027/work/NUCE403/endfb-viii/cross_sections.xml"

import math
import openmc
from openmc.model import borated_water


## Defining materials
t_ref = 600 # Kelvin

fuel = openmc.Material(name='uo2')
fuel.add_nuclide('U235', 0.04)
fuel.add_nuclide('U238', 0.96)
fuel.add_nuclide('O16', 2)
fuel.set_density('g/cm3', 11.5)
fuel.temperature = t_ref

cladding = openmc.Material(name='zircaloy4')
cladding.add_element('Zr',0.98)
cladding.add_element('Sn',0.015)
cladding.add_element('Fe',0.002)
cladding.add_element('Cr',0.001)
cladding.add_element('O',0.001)
cladding.add_element('Hf', 0.001)
cladding.set_density('g/cm3', 6.34)
cladding.temperature = t_ref

#water = openmc.Material(name='h2o')
#water.add_nuclide('H1', 2.0)
#water.add_nuclide('O16', 1.0)
#water.set_density('g/cm3', 1.0)
#water.add_s_alpha_beta('c_H_in_H2O')

boron_ppm_concentration = 1000
temperature_k = t_ref
pressure_mpa = 15 # MPa

water_material = openmc.model.borated_water(
    boron_ppm=boron_ppm_concentration,
    temperature=temperature_k,
    pressure=pressure_mpa
)
water_material.add_s_alpha_beta('c_H_in_H2O')

ctrl_rod = openmc.Material(name='b4c')
ctrl_rod.add_nuclide('B10', 2.8)
ctrl_rod.add_nuclide('B11', 1.2)
ctrl_rod.add_nuclide('C12', 1)
ctrl_rod.set_density('g/cm3', 2.5)
ctrl_rod.temperature = t_ref

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
lat.pitch = (2.35,)           # i picked this number just to make it fit, no calculations.
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
settings.batches = 150
settings.inactive = 30
settings.particles = 10000
settings.run_mode = 'eigenvalue'
settings.export_to_xml()

openmc.run()



