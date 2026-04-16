# ABL Test Case: Shear-Convective Boundary Layer

**Date:** April 8, 2025  
**Authors:** Donati L. and Huusko L.

## Overview
This test case is designed for Atmospheric Boundary Layer (ABL) validation in **Neko**. It is based on the [shear_convection_abl](https://github.com/ExtremeFLOW/neko/tree/develop/examples/shear_convection_abl) example with some minor modifications, namely:
- a passive scalar has been implemented (same i.c. and b.c. of temperatue in the `shear_convection_abl` example)
- temperature now has a time-varying Dirichlet condition as the bottom b.c., which should 
lead to the same temperature evolution as it does with the typical Neumann b.c. in the
`shear_convection_abl` case.

### Performance 
* **Compute Resource:** 1 Dardel CPU node (128 cores).
* **Runtime:** Should complete in less than one hour.

---

## Setup

| Parameter | Configuration |
| :--- | :--- |
| **Type** | Shear-convective ABL |
| **Domain Size** | 5 x 5 x 2 km |
| **Mesh Resolution** | 20 elements per side |
| **Polynomial Degree** | 7 |
| **Boundary Conditions** | Heat flux, Wall model, Geostrophic wind |
| **SGS Model** | TKE model (Deardorff) |
| **Wall Model (WM)** | MOST (Monin-Obukhov Similarity Theory) with `z0 = z0h`|
| **Source Terms** | Coriolis + Boussinesq + Sponge |
| **Sim. Components** | Basic fluid and scalar statistics |

---

## Testing & Validation
Validation is performed via the Python script `compare_ref.py` through three specific checks:

### 1. Statistics Test (`stats_test`)
Compares the generated `.csv` files for fluid and temperature against `ref/fluid_stats_ref.csv`, `ref/temperature_stats_ref.csv` and `ref/scalar_stats_ref.csv`.
* **Failure Condition:** Max % difference > 5% in any variable compared to reference.

### 2. Snapshot Test (`snap_test`)
Takes the last simulation snapshot, performs a horizontal average, and compares it to `ref/snap_ref.f0`. `ref/snap_mesh.f0` is necessary to open the field files.
* **Failure Condition:** Max % difference > 10% in any variable compared to reference.

### 3. Passive vs Active Scalar Test (`temp_test`)
Takes the last simulation snapshot, performs a horizontal average, and compares temperature to 
the passive scalar values at `z=0` (the passive scalar and temperature B.C.'s have been constructed such taht they should give the same surface value).
* **Failure Condition:** Max % difference > 2% in any variable compared to reference.