import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker


class OceanCurrentPlotter:
    def __init__(self):
        self.figsize = (14, 7)
        self.fontsize = 14
        self.labelsize = 12
        self.quiver_scale = 30
        self.skip = 45         # 箭头稀疏程度，可调大

        self.cmap = 'pink_r'   # SCI 安全色表

    def custom_cmap(self):
        colors = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
        return mcolors.LinearSegmentedColormap.from_list("ocean_blue", colors)

    def load_data(self, path):
        ds = xr.open_dataset(path)

        u = ds['water_u'].isel(time=0, depth=0)
        v = ds['water_v'].isel(time=0, depth=0)
        lon = ds['lon']
        lat = ds['lat']

        # 将经度从 0~360 滚动到 -180~180，并排序
        lon_180_idx = (lon >= 180).argmax(dim='lon').item()
        u_rolled = u.roll(lon=-lon_180_idx, roll_coords=True)
        v_rolled = v.roll(lon=-lon_180_idx, roll_coords=True)
        lon_rolled = lon.roll(lon=-lon_180_idx, roll_coords=True)

        lon_adjusted = xr.where(lon_rolled >= 180, lon_rolled - 360, lon_rolled)
        lon_adjusted.attrs = lon_rolled.attrs

        u_rolled = u_rolled.assign_coords(lon=lon_adjusted)
        v_rolled = v_rolled.assign_coords(lon=lon_adjusted)

        speed = np.sqrt(u_rolled**2 + v_rolled**2)

        return lon_adjusted, lat, u_rolled, v_rolled, speed

    def plot(self, lon, lat, u, v, speed):
        proj = ccrs.PlateCarree(central_longitude=180)
        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes(projection=proj)

        ax.set_global()
        ax.coastlines(linewidth=0.8)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)

        # 流速填色
        self.cmap = self.custom_cmap()
        im = ax.pcolormesh(
            lon, lat, speed,
            cmap=self.cmap,
            vmin=0, vmax=1.0,
            shading='auto',
            transform=ccrs.PlateCarree()
        )
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label('Surface Current Speed (m/s)', fontsize=self.fontsize)
        cbar.ax.tick_params(labelsize=self.labelsize)

        # 关键修正：将 DataArray 切片转换为 numpy 数组
        lon_sub = lon[::self.skip].values
        lat_sub = lat[::self.skip].values
        u_sub = u[::self.skip, ::self.skip].values
        v_sub = v[::self.skip, ::self.skip].values

        ax.quiver(
            lon_sub, lat_sub, u_sub, v_sub,
            scale=self.quiver_scale,
            width=0.002,
            color='black',
            transform=ccrs.PlateCarree()
        )

        self.annotate_currents(ax)

        # 经纬度刻度
        xticks = np.arange(-180, 181, 60)
        yticks = np.arange(-90, 91, 30)
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        ax.tick_params(axis='both', labelsize=self.labelsize)

        ax.set_title(
            'Global Surface Ocean Currents (HYCOM 2015)',
            fontsize=self.fontsize + 2
        )

        plt.tight_layout()
        plt.show()

    def annotate_currents(self, ax):
        currents = [
            ("Kuroshio", 140, 30),
            ("Gulf Stream", -60, 35),
            ("Brazil Current", -45, -25),
            ("California Current", -125, 30),
            ("Canary Current", -20, 25),
            ("Agulhas Current", 30, -35),
            ("Antarctic Circumpolar Current", 0, -55),
            ("North Equatorial Current", -140, 10),
            ("South Equatorial Current", -120, -10),
            ("East Australian Current", 155, -30)
        ]
        for name, x, y in currents:
            ax.text(
                x, y, name,
                fontsize=self.labelsize,
                transform=ccrs.PlateCarree(),
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
            )


if __name__ == "__main__":
    path = r"E:\PhysicalOcean\dpo_code\data\GLBv0.08_53X_archMN.2015_01_2015_12_uv3z.nc"
    plotter = OceanCurrentPlotter()
    lon, lat, u, v, speed = plotter.load_data(path)
    plotter.plot(lon, lat, u, v, speed)