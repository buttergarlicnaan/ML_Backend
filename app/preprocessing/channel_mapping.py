"""
Sentinel-2 and WorldStrat channel mappings for Multi-Frame Super-Resolution (MFSR).

The MFSR network expects each temporal observation to have exactly 17 channels:
- Channels 0–11: 12 Sentinel-2 L2A surface reflectance bands
- Channel 12: Valid / Cloud mask (1.0 = clear/valid, 0.0 = cloud/shadow/invalid)
- Channels 13–16: 4 solar and sensor angles

Detailed Channel Breakdown:
--------------------------------------------------------------------------------
Index | Band Name            | Description                         | Expected Range
--------------------------------------------------------------------------------
0     | B01 - Coastal        | Aerosols detection (443 nm)         | [0.0, 1.0] (reflectance)
1     | B02 - Blue           | Blue visible band (490 nm)          | [0.0, 1.0] (reflectance)
2     | B03 - Green          | Green visible band (560 nm)         | [0.0, 1.0] (reflectance)
3     | B04 - Red            | Red visible band (665 nm)           | [0.0, 1.0] (reflectance)
4     | B05 - Red Edge 1     | Vegetation classification (705 nm)  | [0.0, 1.0] (reflectance)
5     | B06 - Red Edge 2     | Vegetation classification (740 nm)  | [0.0, 1.0] (reflectance)
6     | B07 - Red Edge 3     | Vegetation classification (783 nm)  | [0.0, 1.0] (reflectance)
7     | B08 - NIR Broadband  | Biomass & water content (842 nm)    | [0.0, 1.0] (reflectance)
8     | B8A - NIR Narrow     | Atmospheric correction (865 nm)     | [0.0, 1.0] (reflectance)
9     | B09 - Water Vapour   | Water vapour absorption (940 nm)    | [0.0, 1.0] (reflectance)
10    | B11 - SWIR 1         | Snow/cloud discrimination (1610 nm) | [0.0, 1.0] (reflectance)
11    | B12 - SWIR 2         | Soil/geology observation (2190 nm)  | [0.0, 1.0] (reflectance)
12    | CLM / Valid Mask     | Binary/float mask (1=valid, 0=cloud)| {0.0, 1.0}
13    | Sun Zenith           | Solar Zenith Angle                  | Degrees / Radians
14    | Sun Azimuth          | Solar Azimuth Angle                 | Degrees / Radians
15    | View Zenith          | Sensor Viewing Zenith Angle         | Degrees / Radians
16    | View Azimuth         | Sensor Viewing Azimuth Angle        | Degrees / Radians
--------------------------------------------------------------------------------
"""

from typing import Final

EXPECTED_NUM_CHANNELS: Final[int] = 17
NUM_REFLECTANCE_CHANNELS: Final[int] = 12
MASK_CHANNEL_INDEX: Final[int] = 12
NUM_ANGLE_CHANNELS: Final[int] = 4
ANGLE_CHANNEL_START_INDEX: Final[int] = 13

S2_BAND_NAMES: Final[list[str]] = [
    "B01_Coastal_Aerosol",
    "B02_Blue",
    "B03_Green",
    "B04_Red",
    "B05_RedEdge1",
    "B06_RedEdge2",
    "B07_RedEdge3",
    "B08_NIR",
    "B8A_NIRNarrow",
    "B09_WaterVapour",
    "B11_SWIR1",
    "B12_SWIR2",
]

ANGLE_NAMES: Final[list[str]] = [
    "Solar_Zenith",
    "Solar_Azimuth",
    "Sensor_Zenith",
    "Sensor_Azimuth",
]

ALL_17_CHANNEL_NAMES: Final[list[str]] = (
    S2_BAND_NAMES + ["Valid_Cloud_Mask"] + ANGLE_NAMES
)

RGB_INDICES_IN_REFLECTANCE: Final[list[int]] = [3, 2, 1]  # B04 (Red), B03 (Green), B02 (Blue)
RGB_NIR_INDICES_IN_REFLECTANCE: Final[list[int]] = [3, 2, 1, 7]  # Red, Green, Blue, NIR (B08)
