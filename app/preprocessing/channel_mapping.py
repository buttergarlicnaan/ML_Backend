"""
Sentinel-2 and WorldStrat channel mappings for Multi-Frame Super-Resolution (MFSR).
"""

from typing import Final

EXPECTED_NUM_CHANNELS: Final[int] = 17
NUM_REFLECTANCE_CHANNELS: Final[int] = 12
MASK_CHANNEL_INDEX: Final[int] = 12
NUM_ANGLE_CHANNELS: Final[int] = 4
ANGLE_CHANNEL_START_INDEX: Final[int] = 13

S2_BAND_NAMES: Final[list[str]] = [
    'B01_Coastal_Aerosol', 'B02_Blue', 'B03_Green', 'B04_Red',
    'B05_RedEdge1', 'B06_RedEdge2', 'B07_RedEdge3', 'B08_NIR',
    'B8A_NIRNarrow', 'B09_WaterVapour', 'B11_SWIR1', 'B12_SWIR2',
]

ANGLE_NAMES: Final[list[str]] = [
    'Solar_Azimuth',
    'Solar_Zenith',
    'Sensor_Azimuth',
    'Sensor_Zenith',
]

ALL_17_CHANNEL_NAMES: Final[list[str]] = S2_BAND_NAMES + ['Valid_Cloud_Mask'] + ANGLE_NAMES
RGB_INDICES_IN_REFLECTANCE: Final[list[int]] = [3, 2, 1]
RGB_NIR_INDICES_IN_REFLECTANCE: Final[list[int]] = [3, 2, 1, 7]
