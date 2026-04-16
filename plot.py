#%%

import xarray as xr
import neko_utils as nk
import matplotlib.pyplot as plt

# Load data
cases = {
    'ref': '/cfs/klemming/projects/snic/abl-les-ldl/neko_test_case/output',
}

data = {}
for case, path in cases.items():
    data[case] = nk.csv_to_xr(f'{path}/fluid_stats0.csv').mean(dim="time")

# Fluid plots
fluid_vars = [('u', 'v', ), ('uu',), ('vv',), ('ww',), ('uw',), ('w',)]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, vars_to_plot in enumerate(fluid_vars):
    axes[idx].set_title(', '.join(vars_to_plot))
    for v,var in enumerate(vars_to_plot):
        for case, obj in data.items():
            if "no_" in case:
                color, model = "tab:blue", "No ref"
            else:
                color, model = "tab:green", "ref"
            axes[idx].plot(obj[var], obj.z, c=color,label=model)
    axes[idx].legend()

# Temperature data
data_t = {}
for case, path in cases.items():
    data_t[case] = nk.csv_to_xr(
        f'{path}/scalar_stats_temperature0.csv',
        type="scalar", basic=True, height="z",
        fluid_csv=f'{path}/fluid_stats0.csv'
    ).mean(dim="time")

# Temperature plots
temp_vars = [('s',), ('ss',), ('us',), ('vs',), ('ws',)]
temp_titles = ['t', 'tt', 'tu', 'tv', 'tw']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, (var, title) in enumerate(zip(temp_vars, temp_titles)):
    axes[idx].set_title(title)
    for case, obj in data_t.items():
        if "no_" in case:
            color, model = "tab:blue", "No ref"
        else:
            color, model = "tab:green", "ref"
        axes[idx].plot(obj[var[0]], obj.z, c=color,label=model)
    axes[idx].legend()


# Scalar plots
ps_vars = [('s',), ('ss',), ('us',), ('vs',), ('ws',)]
ps_titles = ['s', 'ss', 'su', 'sv', 'sw']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, (var, title) in enumerate(zip(ps_vars, ps_titles)):
    axes[idx].set_title(title)
    for case, obj in data_t.items():
        if "no_" in case:
            color, model = "tab:blue", "No ref"
        else:
            color, model = "tab:green", "ref"
        axes[idx].plot(obj[var[0]], obj.z, c=color,label=model)
    axes[idx].legend()


# %%
