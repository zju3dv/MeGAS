#!/bin/bash
# Melting simulation of the candy house, heated by a spherical heat source.
# Input:  log/gaussians/candy_house/align/align_gaussian.ply
# Output: log/gaussians/candy_house/mpm/solid-jelly_liquid-foam_thermal-sphere/test/@<timestamp>/

SCENE="candy_house"

python gs_simulation.py \
    --gs_file ${SCENE} \
    --solid_material jelly \
    --liquid_material foam \
    --thermal_boundary sphere \
    --thermal_sphere_boundary ${SCENE}.txt \
    --thermal_boundary_value 1.0 \
    --save_note test \
    --use_damping
