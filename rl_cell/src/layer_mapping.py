from enum import Enum


# layer mapping for SkyWater130
class LayerMapping(Enum):
    nwell = (64, 20)
    diff = (65, 20)  # aa
    poly = (66, 20)
    li_ct = (66, 44)
    # li_ct? = (67, 16) # layer (x, 16) might not be a contact / via layer
    li = (67, 20)  # li (local interconnection)
    metal_ct = (67, 44)
    metal1 = (68, 20)
    via1 = (68, 44)
