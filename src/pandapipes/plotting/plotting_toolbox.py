# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import re
import ast
import pandas as pd


def _get_coords_from_geojson(gj_str):
    pattern = r'"coordinates"\s*:\s*((?:\[(?:\[[^]]+],?\s*)+\])|\[[^]]+\])'
    matches = re.findall(pattern, gj_str)

    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError("More than one match found in GeoJSON string")
    for m in matches:
        return ast.literal_eval(m)
    return None


def get_collection_sizes(net, junction_size=1.0, ext_grid_size=1.0, sink_size=1.0, source_size=1.0,
                         valve_size=2.0, pump_size=1.0, heat_exchanger_size=1.0,
                         pressure_control_size=1.0, compressor_size=1.0, flow_control_size=1.0):
    """
    Calculates the size for most collection types according to the distance between min and max
    geocoord so that the collections fit the plot nicely

    .. note: This is implemented because if you would choose a fixed values (e.g.\
        junction_size = 0.2), the size could be to small for large networks and vice versa

    :param net: pandapower network for which to create plot
    :type net: pandapowerNet
    :param junction_size: relative junction size
    :type junction_size: float, default 1.
    :param ext_grid_size: relative external grid size
    :type ext_grid_size: float, default 1.
    :param sink_size: relative sink size
    :type sink_size: float, default 1.
    :param source_size: relative source size
    :type source_size: float, default 1.
    :param valve_size: relative valve size
    :type valve_size: float, default 2.
    :param heat_exchanger_size: relative heat exchanger size
    :type heat_exchanger_size: float, default 1.
    :return: sizes (dict) - dictionary containing all scaled sizes
    """

    coords = pd.DataFrame(net.junction.geo.apply(_get_coords_from_geojson), columns=['x', 'y'])
    mean_distance_between_junctions = sum((coords.max() - coords.min()).dropna() / 200)

    sizes = {
        "junction": junction_size * mean_distance_between_junctions,
        "ext_grid": ext_grid_size * mean_distance_between_junctions * 2,
        "valve": valve_size * mean_distance_between_junctions * 2,
        "sink": sink_size * mean_distance_between_junctions * 2,
        "source": source_size * mean_distance_between_junctions * 2,
        "heat_exchanger": heat_exchanger_size * mean_distance_between_junctions * 8,
        "pump": pump_size * mean_distance_between_junctions * 8,
        "pressure_control": pressure_control_size * mean_distance_between_junctions * 8,
        "compressor": compressor_size * mean_distance_between_junctions * 8,
        "flow_control": flow_control_size * mean_distance_between_junctions * 2,
    }

    return sizes
