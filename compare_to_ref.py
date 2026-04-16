#%%
import sys
import numpy as np
import pandas as pd
import xarray as xr
import pymech as pm
import matplotlib.pyplot as plt

##########################################################################################
#######################################  FUNCTIONS  ######################################
##########################################################################################

def open_neko_dataset(path, mesh_ref):
    """
    Robust loader for Neko fields. 
    Handles GLL point extraction more carefully to avoid xarray size conflicts.
    """
    field = pm.readnek(path)
    mesh_field = pm.readnek(mesh_ref)
    
    if isinstance(field, int) or isinstance(mesh_field, int):
        raise OSError(f"Failed to load: {path} or {mesh_ref}")

    axes = ("z", "y", "x")
    elem_dsets = []

    for e_data, e_geom in zip(field.elem, mesh_field.elem):
        # Extract unique coordinates per element. 
        # We use a finer precision (10) to avoid collapsing GLL points.
        # reverse 'axes' because pymech pos[0,1,2] is x,y,z
        elem_coords = {}
        for i, ax in enumerate(["x", "y", "z"]):
            unq = np.unique(np.round(e_geom.pos[i], 10))
            elem_coords[ax] = unq

        # Ensure coordinate lengths match the data shape (e.g., 8x8x8)
        # If np.unique fails to find the right amount, we generate dummy indices
        # to prevent the 'length 1 vs length 8' error.
        for ax in axes:
            if len(elem_coords[ax]) != e_data.vel[0].shape[axes.index(ax)]:
                # Fallback: Use the first value of the mesh for this element
                # if the coordinates are too close to distinguish
                start_val = e_geom.pos[axes.index(ax)].min()
                end_val = e_geom.pos[axes.index(ax)].max()
                elem_coords[ax] = np.linspace(start_val, end_val, e_data.vel[0].shape[axes.index(ax)])

        data_vars = {
            "ux": (axes, e_data.vel[0]), "uy": (axes, e_data.vel[1]), "uz": (axes, e_data.vel[2]),
            "xmesh": (axes, e_geom.pos[0]), "ymesh": (axes, e_geom.pos[1]), "zmesh": (axes, e_geom.pos[2]),
        }
        if e_data.pres.size: data_vars["pressure"] = (axes, e_data.pres[0])
        if e_data.temp.size: data_vars["temperature"] = (axes, e_data.temp[0])
        
        for i in range(e_data.scal.shape[0]):
            data_vars[f"s{i+1:02d}"] = (axes, e_data.scal[i])

        # Create local dataset for this element
        ds_el = xr.Dataset(data_vars, coords=elem_coords)
        elem_dsets.append(ds_el)

    # Combine all elements. compat='override' ignores tiny coordinate mismatches
    # at element boundaries.
    try:
        ds = xr.combine_by_coords(elem_dsets, compat='override', coords='minimal')
    except Exception as e:
        print(f"Combine failed: {e}. Falling back to a non-aligned merge.")
        ds = xr.merge(elem_dsets, compat='override')

    ds.coords.update({"time": field.time})
    return ds

def snap_test(snap, snap_ref, mesh_ref,threshold=20.0):
    print("\n--- Running snap_test ---")
    try:
        # We load the full field and average it
        ds = open_neko_dataset(snap, mesh_ref)
        ds_ref = open_neko_dataset(snap_ref, mesh_ref)
        
        # Perform horizontal average
        # Using 'xmesh'/'ymesh' names if they are data_vars, 
        # or just dimensions x and y
        mean_test = ds.mean(dim=['x', 'y'])
        mean_ref = ds_ref.mean(dim=['x', 'y'])
        
        vars_to_plot = ["ux", "uy", "uz", "temperature", "s01"]
        fig, axes = plt.subplots(1, 5, figsize=(15, 5))
        passed = True

        for i, var in enumerate(vars_to_plot):
            if var == "temperature":
                err = calc_percent_diff(mean_test[var].values-273.15, mean_ref[var].values-273.15)
            else:
                err = calc_percent_diff(mean_test[var].values, mean_ref[var].values)
            axes[i].plot(mean_test[var], mean_test.z, label='Test', c='tab:blue')
            axes[i].plot(mean_ref[var], mean_ref.z, label='Ref', c='tab:green', ls='--')
            axes[i].set_title(f"{var} (Max Err: {err:.2f}%)")
            axes[i].legend()
            if err > threshold: passed = False
            print(f"{'[PASSED]' if err <= threshold else '[FAILED]'} {var}: {err:.4f}%")
        
        plt.tight_layout()
        plt.show()
        return passed
    except Exception as e:
        print(f"[ERROR] snap_test: {e}")
        return False

def temp_scal_test(snap, mesh_ref, threshold=5.0):
    print("\n--- Running temp_test ---")
    try:
        ds = open_neko_dataset(snap, mesh_ref)
        mean = ds.mean(dim=['x', 'y'])
        
        # Check variable names. Neko uses 'temperature' and scalars like 's01'
        t_var = 'temperature'
        s_var = 's01'
        
        t_prof = mean[t_var]
        s_prof = mean[s_var]
        
        err = calc_percent_diff(t_prof.values-273.15, s_prof.values-273.15)
        
        plt.figure(figsize=(5, 7))
        plt.plot(t_prof, mean.z, label='Active Temp', c='red')
        plt.plot(s_prof, mean.z, label='Passive Scalar', c='blue', ls='--')
        plt.title(f"Temp Comparison (Max Diff: {err:.2f}%)")
        plt.legend()
        plt.show()
        
        print(f"{'[PASSED]' if err <= threshold else '[FAILED]'} Temp vs Scalar: {err:.4f}%")
        return err <= threshold
    except Exception as e:
        print(f"[ERROR] temp_test: {e}")
        return False

def csv_to_xr(path, type="fluid", height="z", fluid_csv=None):
    vars_map = {
        "fluid": ["p", "u", "v", "w", "pp", "uu", "vv", "ww", "uv", "uw", "vw"],
        "scalar": ["s", "us", "vs", "ws", "ss"]
    }
    v_list = vars_map[type]
    df = pd.read_csv(path, header=None, names=['time', height] + v_list)
    df[height] = df[height].round(6)
    ds = df.groupby(['time', height]).mean().to_xarray()
    if type == "scalar":
        if not fluid_csv: raise ValueError("Scalar type requires fluid_csv.")
        f_ds = csv_to_xr(fluid_csv, type="fluid", height=height)
        for ax in ["u", "v", "w"]:
            if f"{ax}s" in ds:
                ds[f"{ax}s"] = ds[f"{ax}s"] - (f_ds[ax] * ds["s"])
        ds["ss"] = ds["ss"] - (ds["s"]**2)
    else:
        for var in ["uu", "vv", "ww", "uv", "uw", "vw"]:
            ds[var] = ds[var] - (ds[var[0]] * ds[var[1]])
    return ds

def calc_percent_diff(test_val, ref_val):
    if np.allclose(test_val, ref_val, atol=1e-12): return 0.0
    diff = np.abs(test_val - ref_val)
    denom = np.abs(ref_val) + 1e-10
    return np.nanmax(diff / denom) * 100

def stats_test(f_csv, f_ref, t_csv, t_ref, s_csv, s_ref, height="z", threshold=1.0):
    print("--- Running stats_test ---")
    passed = True
    try:
        ds_f = csv_to_xr(f_csv, "fluid", height=height)
        ds_f_ref = csv_to_xr(f_ref, "fluid", height=height)
        ds_t = csv_to_xr(t_csv, "scalar", fluid_csv=f_csv, height=height)
        ds_t_ref = csv_to_xr(t_ref, "scalar", fluid_csv=f_ref, height=height)
        ds_s = csv_to_xr(s_csv, "scalar", fluid_csv=f_csv, height=height)
        ds_s_ref = csv_to_xr(s_ref, "scalar", fluid_csv=f_ref, height=height)

        # Plot Fluid Stats
        fluid_plots = [('u', 'v', 'w'), ('uu', 'vv', 'ww'), ('uv', 'uw', 'vw')]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Fluid Statistics Comparison')
        for i, vars_to_plot in enumerate(fluid_plots):
            for var in vars_to_plot:
                axes[i].plot(ds_f[var].mean(dim="time"), ds_f[height], label=f'Test {var}', linestyle='-')

                axes[i].plot(ds_f_ref[var].mean(dim="time"), ds_f_ref[height], label=f'Ref {var}', linestyle='--')
            axes[i].set_title(f'Variables: {", ".join(vars_to_plot)}')
            axes[i].legend()
        fig.savefig(f'fluid_stats_comparison.png', dpi=300, bbox_inches='tight')

        # Plot Temperature Stats
        scalar_vars = ['s', 'ss', 'us', 'vs', 'ws']
        fig1, axes1 = plt.subplots(1, 5, figsize=(20, 5))
        fig1.suptitle('Temperature Statistics Comparison')
        for i, var in enumerate(scalar_vars):
            axes1[i].plot(ds_t[var].mean(dim="time"), ds_t[height], label='Test', linestyle='-')
            axes1[i].plot(ds_t_ref[var].mean(dim="time"), ds_t_ref[height], label='Ref', linestyle='--')
            axes1[i].set_title(var)
            axes1[i].legend()
        fig1.savefig(f'temperature_stats_comparison.png', dpi=300, bbox_inches='tight')

        # Plot Scalar Stats
        scalar_vars = ['s', 'ss', 'us', 'vs', 'ws']
        fig2, axes2 = plt.subplots(1, 5, figsize=(20, 5))
        fig2.suptitle('Scalar Statistics Comparison')
        for i, var in enumerate(scalar_vars):
            axes2[i].plot(ds_s[var].mean(dim="time"), ds_s[height], label='Test', linestyle='-')
            axes2[i].plot(ds_s_ref[var].mean(dim="time"), ds_s_ref[height], label='Ref', linestyle='--')
            axes2[i].set_title(var)
            axes2[i].legend()
        fig2.savefig(f'scalar_stats_comparison.png', dpi=300, bbox_inches='tight')

        for ds_x, ds_r, label in [(ds_f, ds_f_ref, "Fluid"), (ds_t, ds_t_ref, "Temperature"), (ds_s, ds_s_ref, "Scalar")]:
            for var in ds_x.data_vars:
                err = calc_percent_diff(ds_x[var].values, ds_r[var].values)
                status = "[PASSED]" if err <= threshold else "[FAILED]"
                print(f"{status} {label} - {var}: {err:.4f}%")
                if err > threshold: passed = False
    except Exception as e:
        print(f"[ERROR] stats_test: {e}"); passed = False
    return passed


##########################################################################################
#########################################  MAIN  #########################################
##########################################################################################

#%%
if __name__ == "__main__":
    # Ensure these paths point to your actual local files
    FLUID_STAT      = "output/fluid_stats0.csv"
    FLUID_STAT_REF  = "ref/fluid_stats_ref.csv"
    TEMP_STAT       = "output/scalar_stats_temperature0.csv"
    TEMP_STAT_REF   = "ref/temperature_stats_ref.csv"
    SCALAR_STAT     = "output/scalar_stats_s010.csv"
    SCALAR_STAT_REF = "ref/scalar_stats_ref.csv"

    SNAP     = "output/field0.f00005"
    SNAP_REF = "ref/snap_ref.f0"

    MESH = "ref/snap_mesh.f0"

    print("Starting validation and plotting...\n")
    
    t1 = stats_test(FLUID_STAT, FLUID_STAT_REF, 
                    TEMP_STAT, TEMP_STAT_REF,
                    SCALAR_STAT, SCALAR_STAT_REF,
                    threshold=5)
    t2 = snap_test(SNAP, SNAP_REF, MESH,
                   threshold=10)
    t3 = temp_scal_test(SNAP, MESH,
                   threshold=2)

    print("\n==================================")
    if all([t1, t2, t3]):
        print("OVERALL RESULT: ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("OVERALL RESULT: ONE OR MORE TESTS FAILED")
        sys.exit(1)

#%%