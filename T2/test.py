import xarray as xr

path = 'E:\PhysicalOcean\dpo_code\data\GLBv0.08_53X_archMN.2015_01_2015_12_uv3z.nc'
ds = xr.open_dataset(path)

print(ds)
