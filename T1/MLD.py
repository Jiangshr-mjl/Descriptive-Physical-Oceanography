import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import re


class MLDCalculator:
    def __init__(self, temp_threshold=0.2, ref_depth_index=None):
        """
        初始化混合层计算器
        :param temp_threshold: 温度阈值（度）
        :param ref_depth_index: 参考深度在深度列中的索引，默认为0（即最浅层）
        """
        self.temp_threshold = temp_threshold
        self.ref_depth_index = ref_depth_index if ref_depth_index is not None else 0

        self.fontsize = 15
        self.labelsize = 14

    @staticmethod
    def _parse_depth_columns(columns):
        """
        从列名中解析出深度值（米）
        假设深度列名格式为 '0m'、'5m'、... 或 'depth0'、'depth5' 等，
        这里统一提取数字部分作为深度。
        """
        depths = []
        depth_cols = []
        for col in columns:
            # 匹配数字（整型或浮点）后可能跟'm'或无单位
            match = re.search(r'(\d+\.?\d*)\s*m?', str(col))
            if match:
                depths.append(float(match.group(1)))
                depth_cols.append(col)
            else:
                # 若不是深度列（如Latitude、Longitude），则忽略
                continue
        # 按深度从小到大排序
        sorted_indices = np.argsort(depths)
        depths = [depths[i] for i in sorted_indices]
        depth_cols = [depth_cols[i] for i in sorted_indices]
        return depths, depth_cols

    def load_data(self, filepath):
        """
        读取WOA CSV数据文件
        :param filepath: CSV文件路径
        :return: DataFrame（含Latitude, Longitude, 各深度层温度）
        """
        # 跳过第一行，第二行作为列名
        df = pd.read_csv(filepath, header=1)
        # 重命名前两列为'Latitude'和'Longitude'，防止空格等问题
        df.rename(columns={df.columns[0]: 'Latitude', df.columns[1]: 'Longitude'}, inplace=True)
        return df

    def compute_mld(self, df):
        """
        计算混合层深度
        :param df: 包含温度和地理信息的DataFrame
        :return: DataFrame，包含经度、纬度、MLD(m)
        """
        # 提取深度列及对应温度数据
        columns = df.columns.tolist()
        # 去除前两列（地理信息）
        data_columns = columns[2:]
        depths, depth_cols = self._parse_depth_columns(data_columns)

        # 确保深度列顺序与原始列一致
        depth_array = np.array(depths)
        temp_array = df[depth_cols].values  # shape: (n_points, n_depths)

        # 参考温度（最浅层）
        ref_temp = temp_array[:, self.ref_depth_index]

        # 计算混合层深度
        mld = np.full(len(df), np.nan)
        for i in range(len(df)):
            # 从浅到深检查
            for j, d in enumerate(depth_array):
                # 如果 j <= self.ref_depth_index 跳过（只需在参考层之下查找）
                if j <= self.ref_depth_index:
                    continue
                # 温度差绝对值
                diff = abs(temp_array[i, j] - ref_temp[i])
                if diff > self.temp_threshold:
                    # 取当前层与上一层之间的深度（线性插值更精确，此处简单取当前层深度）
                    # 也可取上一层深度，或平均。这里用depth_array[j]作为混合层深度
                    mld[i] = depth_array[j]
                    break

        # 构建结果DataFrame
        result = pd.DataFrame({
            'Longitude': df['Longitude'],
            'Latitude': df['Latitude'],
            'MLD': mld
        })
        return result

    def plot_mld(self, mld_winter, mld_summer, output_path='global_mld.png'):
        """
        绘制冬季和夏季全球混合层深度分布
        :param mld_winter: 冬季MLD DataFrame
        :param mld_summer: 夏季MLD DataFrame
        :param output_path: 输出图像路径
        """
        # 确定统一的colormap和norm
        # vmin = min(mld_winter['MLD'].min(), mld_summer['MLD'].min())
        # vmax = max(mld_winter['MLD'].max(), mld_summer['MLD'].max())
        vmin = 10
        vmax = 500

        fig, axes = plt.subplots(2, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        cmap = plt.get_cmap('jet')

        # 冬季
        ax = axes[0]
        sc = ax.scatter(mld_winter['Longitude'], mld_winter['Latitude'],
                        c=mld_winter['MLD'], cmap=cmap, vmin=vmin, vmax=vmax,
                        s=1, marker='s', transform=ccrs.PlateCarree())
        ax.set_title('Winter Mixed Layer Depth (m)', fontsize=self.fontsize)
        ax.coastlines()
        ax.set_global()
        ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(axis='both', which='major', labelsize=self.labelsize)

        # 夏季
        ax = axes[1]
        sc = ax.scatter(mld_summer['Longitude'], mld_summer['Latitude'],
                        c=mld_summer['MLD'], cmap=cmap, vmin=vmin, vmax=vmax,
                        s=1, marker='s', transform=ccrs.PlateCarree())
        ax.set_title('Summer Mixed Layer Depth (m)', fontsize=self.fontsize)
        ax.coastlines()
        ax.set_global()
        ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(axis='both', which='major', labelsize=self.labelsize)

        # 公用colorbar
        cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.1, pad=0.04, extend='max')
        cbar.set_label('Mixed Layer Depth (m)', fontsize=self.fontsize)
        cbar.ax.tick_params(labelsize=self.labelsize)

        plt.savefig(output_path, dpi=200)
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 初始化计算器，温度阈值0.2°C
    calculator = MLDCalculator(temp_threshold=0.2)

    # 替换为你的实际文件路径
    winter_file = "D:\PhysicalOcean\dpo_code\data\woa18_decav_t13mn01_winter.csv"
    summer_file = "D:\PhysicalOcean\dpo_code\data\woa18_decav_t15mn01_summer.csv"

    # 读取数据
    df_winter = calculator.load_data(winter_file)
    df_summer = calculator.load_data(summer_file)

    # 计算混合层深度
    mld_winter = calculator.compute_mld(df_winter)
    mld_summer = calculator.compute_mld(df_summer)

    # 绘制混合层深度分布图
    calculator.plot_mld(mld_winter, mld_summer, output_path="global_mld_distribution.png")
