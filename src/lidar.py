import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import src.__config as  GLOBALS


class Lidar:
    def __init__(self):

        # Lidar
        self.num_beams: int  = GLOBALS.LIDAR_NUM_BEAMS
        self.angle_min: int  = GLOBALS.LIDAR_ANGLE_MIN
        self.angle_max: int  = GLOBALS.LIDAR_ANGLE_MAX
        self.angle_step: int = GLOBALS.LIDAR_ANGLE_STEP

        # Data
        self.path_data: str = GLOBALS.PATH_DATASET
        self.data_df: pd.DataFrame = self._load_df()


    # Getters
    def get_odometry(self, scan_idx: int) -> tuple[float, float]:
        """Get odometry measurements for a specific scan"""

        return (
            self.data_df.loc[scan_idx, 'delta_s'],
            self.data_df.loc[scan_idx, 'delta_theta']
        )
    
    def get_ranges(self, scan_idx: int) -> np.ndarray:
        """Get LIDAR range measurements for a specific scan"""

        range_columns = [f'angle_{i}' for i in range(self.angle_min, self.angle_max + 1)]
        return self.data_df.loc[scan_idx, range_columns].values
    
    def get_angles(self) -> np.ndarray:
        """Get an array of all the angles"""

        return np.asarray([i for i in range(self.angle_min, self.angle_max+1)])
    
    def get_num_scans(self) -> int:
        """Get total number of scans in the dataset"""

        return len(self.data_df)

    # Helpers
    def _load_df(self):

        data = pd.read_csv(
                self.path_data,
                sep=r'\s+',
                header=None,
                engine='python'
            )

        column_names: list[str] = ['delta_s', 'delta_theta'] + [f'angle_{i}' for i in range(self.angle_min, self.angle_max+1)]

        data.columns = column_names

        return data
        
    def viz_dataframe(self):

        print(self.data_df.head())




if __name__ == "__main__":
    lidar: Lidar = Lidar()
    
    lidar.viz_dataframe()