# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import re
import ast
import numpy as np
import pandas as pd
from pandapower.plotting.collections import _create_node_collection, \
    _create_node_element_collection, _create_line2d_collection, _create_complex_branch_collection, \
    add_cmap_to_collection, coords_from_node_geodata
from pandapower.plotting.patch_makers import load_patches, ext_grid_patches
from pandas import Series

from pandapipes.plotting.patch_makers import valve_patches, source_patches, heat_exchanger_patches, \
    pump_patches, pressure_control_patches, compressor_patches, flow_control_patches
from pandapower.plotting.plotting_toolbox import get_index_array

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


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


def create_junction_collection(net, junctions=None, size=5, patch_type="circle", color=None,
                               z=None, cmap=None, norm=None, infofunc=None, picker=False,
                               junction_geodata=None, cbar_title="Junction Pressure [bar]",
                               **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes junctions.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param junctions: The junctions for which the collections are created.
                      If None, all junctions in the network are considered.
    :type junctions: list, default None
    :param size: Patch size
    :type size: int, default 5
    :param patch_type: Patch type, can be \n
        - "circle" or "ellipse" for an ellipse (cirlces are just ellipses with the same width +\
            height)
        - "rect" or "rectangle" for a rectangle
        - "poly<n>" for a polygon with n edges
    :type patch_type: str, default "circle"
    :param color: Color or list of colors for every element
    :type color: iterable, float, default None
    :param z: Array of magnitudes for colormap. Used in case of given cmap. If None,\
        net.res_junction.p_bar is used.
    :type z: array, default None
    :param cmap: colormap for the patch colors
    :type cmap: matplotlib colormap object, default None
    :param norm:  matplotlib norm object
    :type norm: matplotlib norm object, default None
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param picker: Picker argument passed to the patch collection
    :type picker: bool, default False
    :param junction_geodata: Coordinates to use for plotting. If None, net["junction_geodata"] is\
        used
    :type junction_geodata: pandas.DataFrame, default None
    :param cbar_title: colormap bar title in case of given cmap
    :type cbar_title: str, default "Junction Pressure [bar]"
    :param kwargs: Keyword arguments are passed to the patch function and the patch maker
    :return: pc (matplotlib collection object) - patch collection
    """
    junctions = get_index_array(junctions, net.junction.index)
    if len(junctions) == 0:
        return None

    if any(net.junction.geo.isna()):
        raise AttributeError('net.junction.geo contains NaN values, consider dropping them beforehand.')

    if junction_geodata is None:
        junction_geodata = net.junction.geo.apply(_get_coords_from_geojson)

    junctions_with_geo = junctions[np.isin(junctions, junction_geodata.index.values)]
    if len(junctions_with_geo) < len(junctions):
        logger.warning(
            f"The following junctions cannot be displayed, as there is no geodata available: {(set(junctions) - set(junctions_with_geo))}"
        )

    coords = junction_geodata.loc[junctions_with_geo].values

    infos = [infofunc(junc) for junc in junctions_with_geo] if infofunc is not None else []

    pc = _create_node_collection(junctions_with_geo, coords, size, patch_type, color, picker, infos,
                                 **kwargs)

    if cmap is not None:
        if z is None:
            z = net.res_junction.p_bar.loc[junctions_with_geo]
        add_cmap_to_collection(pc, cmap, norm, z, cbar_title)

    return pc


def create_pipe_collection(net, pipes=None, pipe_geodata=None, junction_geodata=None,
                           use_junction_geodata=False, infofunc=None, cmap=None, norm=None,
                           picker=False, z=None, cbar_title="Pipe Loading [%]", clim=None,
                           **kwargs):
    """
    Creates a matplotlib pipe collection of pandapipes pipes.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param pipes: The pipes for which the collections are created. If None, all pipes
            in the network are considered.
    :type pipes: list, default None
    :param pipe_geodata: Coordinates to use for plotting. If None, net.pipe["geo"] is used.
    :type pipe_geodata: pandas.DataFrame, default None
    :param junction_geodata: Coordinates to use for plotting in case of use_junction_geodata=True.\
        If None, net.junction["geo"] is used.
    :type junction_geodata: pandas.DataFrame, default None
    :param use_junction_geodata: Defines whether junction or pipe geodata are used.
    :type use_junction_geodata: bool, default False
    :param infofunc: infofunction for the line element
    :type infofunc: function, default None
    :param cmap: colormap for the line colors
    :type cmap: matplotlib norm object, default None
    :param norm: matplotlib norm object
    :type norm: matplotlib norm object, default None
    :param picker: Picker argument passed to the patch collection
    :type picker: bool, default False
    :param z: Array of pipe loading magnitudes for colormap. Used in case of given cmap. If None,\
        net.res_pipe.loading_percent is used.
    :type z: array, default None
    :param cbar_title: colormap bar title in case of given cmap
    :type cbar_title: str, default "Pipe Loading [%]"
    :param clim: Setting the norm limits for image scaling
    :type clim: tuple of floats, default None
    :param kwargs: Keyword arguments are passed to the patch function and the patch maker
    :return: lc (matplotlib line collection) - line collection for pipes
    """

    if (use_junction_geodata is False and
            pipe_geodata is None and
            ("geo" not in net.pipe.columns or net.pipe.geo.isnull().all())):
        # if bus geodata is available, but no line geodata
        logger.warning("use_junction_geodata is automatically set to True, since net.pipe.geo is empty.")
        use_junction_geodata = True

    pipes = get_index_array(pipes, net.pipe.index)
    if len(pipes) == 0:
        return None

    pipe_geodata: Series[str] = pipe_geodata.loc[pipes] if pipe_geodata is not None else net.pipe.geo.loc[pipes]
    pipes_without_geo = pipe_geodata.index[pipe_geodata.isna()]

    if use_junction_geodata or not pipes_without_geo.empty:
        elem_indices = pipes if use_junction_geodata else pipes_without_geo
        geos, pipes_with_geo = coords_from_node_geodata(
            element_indices=elem_indices,
            from_nodes=net.pipe.loc[elem_indices, 'from_junction'].values,
            to_nodes=net.pipe.loc[pipes, 'to_junction'].values,
            node_geodata=junction_geodata if junction_geodata is not None else net.junction.geo,
            table_name="pipe",
            node_name="Junction"
        )

        pipe_geodata = pd.Series(geos, index=pipes_with_geo).combine_first(pipe_geodata)

    pipes_without_geo = pipe_geodata.index[pipe_geodata.isna()]
    if not pipes_without_geo.empty:
        logger.warning(f"Could not plot pipes {pipes_without_geo}. Junction geodata is missing for those pipes!")

    infos = [infofunc(pipe) for pipe in pipe_geodata] if infofunc else []

    coords = [_get_coords_from_geojson(pipe_gj) for pipe_gj in pipe_geodata]

    lc = _create_line2d_collection(coords, pipe_geodata.index, infos, picker, **kwargs)

    if cmap is not None:
        if z is None:
            z = net.res_pipe.v_mean_m_per_s.loc[pipe_geodata.index]
        elif isinstance(z, pd.Series):
            z = z.loc[pipe_geodata.index]
        add_cmap_to_collection(lc, cmap, norm, z, cbar_title, clim=clim)

    return lc


def create_sink_collection(net, sinks=None, size=1., infofunc=None, picker=False,
                           orientation=(np.pi*5/6), cmap=None, norm=None, z=None, **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes sinks.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param sinks: The sinks for which the collections are created. If None, all sinks
                  connected to junctions that have junction_geodata entries are considered.
    :type sinks: list, default None
    :param size: Patch size
    :type size: float, default 1
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param picker: Picker argument passed to the patch collection
    :type picker: bool, default False
    :param orientation: Orientation of sink collection. pi is directed downwards, increasing values\
        lead to clockwise direction changes.
    :type orientation: float, default np.pi*(5/6)
    :param cmap: colormap for the sink colors
    :type cmap: matplotlib norm object, default None
    :param norm: matplotlib norm object to normalize the values of z
    :type norm: matplotlib norm object, default None
    :param z: Array of sink result magnitudes for colormap. Used in case of given cmap. If None,\
        net.res_sink.mdot_kg_per_s is used.
    :type z: array, default None
    :param kwargs: Keyword arguments are passed to the patch function
    :return: sink_pc - patch collection, sink_lc - line collection
    """
    sinks = get_index_array(sinks, net.sink.index)
    if len(sinks) == 0:
        return None
    infos = [infofunc(i) for i in range(len(sinks))] if infofunc is not None else []
    node_coords = net.junction.geo.loc[net.sink.loc[sinks, "junction"].values].apply(_get_coords_from_geojson)

    colors = kwargs.pop("color", "k")
    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)
    if cmap is not None:
        if z is None:
            z = net.res_sink.mdot_kg_per_s
        colors = [cmap(norm(z.at[idx])) for idx in sinks]
    patch_edgecolor = kwargs.pop("patch_edgecolor", colors)
    line_color = kwargs.pop("line_color", colors)

    sink_pc, sink_lc = _create_node_element_collection(
        node_coords, load_patches, size=size, infos=infos, orientation=orientation,
        picker=picker, patch_edgecolor=patch_edgecolor, line_color=line_color,
        linewidths=linewidths, **kwargs)
    return sink_pc, sink_lc


def create_source_collection(net, sources=None, size=1., infofunc=None, picker=False,
                             orientation=(np.pi*7/6), cmap=None, norm=None, z=None, **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes sources.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param sources: The sources for which the collections are created. If None, all sources
                    connected to junctions that have junction_geodata entries are considered.
    :type sources: list, default None
    :param size: Patch size
    :type size: float, default 1.
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param picker: Picker argument passed to the patch collection
    :type picker: bool, default False
    :param orientation: Orientation of source collection. pi is directed downwards, increasing\
        values lead to clockwise direction changes.
    :type orientation: float, default np.pi*(7/6)
    :param cmap: colormap for the source colors
    :type cmap: matplotlib norm object, default None
    :param norm: matplotlib norm object to normalize the values of z
    :type norm: matplotlib norm object, default None
    :param z: Array of source result magnitudes for colormap. Used in case of given cmap. If None,\
        net.res_source.mdot_kg_per_s is used.
    :type z: array, default None
    :param kwargs: Keyword arguments are passed to the patch function
    :return: source_pc - patch collection, source_lc - line collection
    """
    sources = get_index_array(sources, net.source.index)
    if len(sources) == 0:
        return None
    infos = [infofunc(i) for i in range(len(sources))] if infofunc is not None else []
    node_coords = net.junction.geo.loc[net.source.loc[sources, "junction"]].apply(_get_coords_from_geojson)

    colors = kwargs.pop("color", "k")
    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)
    if cmap is not None:
        if z is None:
            z = net.res_source.mdot_kg_per_s
        colors = [cmap(norm(z.at[idx])) for idx in sources]
    patch_edgecolor = kwargs.pop("patch_edgecolor", colors)
    line_color = kwargs.pop("line_color", colors)

    source_pc, source_lc = _create_node_element_collection(
        node_coords, source_patches, size=size, infos=infos, orientation=orientation,
        picker=picker, patch_edgecolor=patch_edgecolor, line_color=line_color,
        linewidths=linewidths, repeat_infos=(1, 3), **kwargs)
    return source_pc, source_lc


def create_ext_grid_collection(net, size=1., infofunc=None, orientation=0, picker=False,
                               ext_grids=None, ext_grid_junctions=None, **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes ext_grid. Parameters
    ext_grids, ext_grid_junctions can be used to specify, which ext_grids the collection should be
    created for.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param size: Patch size
    :type size: float, default 1.
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param orientation: Orientation of ext_grid collection. 0 is directed upwards,
                        increasing values lead to clockwise direction changes.
    :type orientation: float, default 0
    :param picker: Picker argument passed to the patch collection
    :type picker: bool, default False
    :param ext_grids: The ext_grids for which the collections are created. If None, all ext_grids
                      which have the entry coords in ext_grid_geodata are considered.
    :type ext_grids: list, default None
    :param ext_grid_junctions: Junctions to be used as ext_grid locations
    :type ext_grid_junctions: np.ndarray, default None
    :param kwargs: Keyword arguments are passed to the patch function
    :return: ext_grid1 - patch collection, ext_grid2 - patch collection

    """
    ext_grids = get_index_array(ext_grids, net.ext_grid.index)
    if ext_grid_junctions is None:
        ext_grid_junctions = net.ext_grid.junction.loc[ext_grids].values
    else:
        if len(ext_grids) != len(ext_grid_junctions):
            raise ValueError("Length mismatch between chosen ext_grids and ext_grid_junctions.")
    infos = [infofunc(ext_grid_idx) for ext_grid_idx in ext_grids] if infofunc is not None else []

    colors = kwargs.pop("color", "k")
    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)
    patch_edgecolor = kwargs.pop("patch_edgecolor", colors)
    line_color = kwargs.pop("line_color", colors)

    node_coords = net.junction.geo.loc[ext_grid_junctions].apply(_get_coords_from_geojson).values
    ext_grid_pc, ext_grid_lc = _create_node_element_collection(
        node_coords, ext_grid_patches, size=size, infos=infos, orientation=orientation,
        picker=picker, hatch="XXX", patch_edgecolor=patch_edgecolor, line_color=line_color,
        linewidths=linewidths, **kwargs)
    return ext_grid_pc, ext_grid_lc


def create_heat_exchanger_collection(net, heat_ex=None, size=5., junction_geodata=None,
                                     infofunc=None, picker=False, **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes junction-junction heat_exchangers.
    Heat_exchangers are plotted in the center between two junctions with a "helper" line
    (dashed and thin) being drawn  between the junctions as well.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param size: Patch size
    :type size: float, default 2.
    :param helper_line_style: Line style of the "helper" line being plotted between two junctions
                                connected by a junction-junction heat_exchanger
    :type helper_line_style: str, default ":"
    :param helper_line_size: Line width of the "helper" line being plotted between two junctions
                                connected by a junction-junction heat_exchanger
    :type helper_line_size: float, default 1.
    :param helper_line_color: Line color of the "helper" line being plotted between two junctions
                                connected by a junction-junction valve
    :type helper_line_color: str, default "gray"
    :param orientation: Orientation of heat_exchanger collection. pi is directed downwards,
                        increasing values lead to clockwise direction changes.
    :type orientation: float, default np.pi/2
    :param kwargs: Keyword arguments are passed to the patch function
    :return: heat_exchanger, helper_lines
    :rtype: tuple of patch collections
    """
    heat_ex = get_index_array(heat_ex, net.heat_exchanger.index)
    hex_table = net.heat_exchanger.loc[heat_ex]

    geos, hex_with_geo = coords_from_node_geodata(
        heat_ex, hex_table.from_junction.values, hex_table.to_junction.values,
        junction_geodata if junction_geodata is not None else net.junction.geo,
        "heat_exchanger", "Junction")
    coords = [_get_coords_from_geojson(geo) for geo in geos]

    if len(hex_with_geo) == 0:
        return None

    colors = kwargs.pop("color", "k")
    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)
    patch_edgecolor = kwargs.pop("patch_edgecolor", colors)
    line_color = kwargs.pop("line_color", colors)

    infos = list(np.repeat([infofunc(i) for i in range(len(hex_with_geo))], 2)) \
        if infofunc is not None else []

    pc, lc = _create_complex_branch_collection(
        coords, heat_exchanger_patches, size, infos, picker=picker, linewidths=linewidths,
        patch_edgecolor=patch_edgecolor, line_color=line_color, **kwargs)

    return pc, lc


def create_valve_collection(net, valves=None, size=5., junction_geodata=None, infofunc=None,
                            picker=False, fill_closed=True, respect_valves=False, **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes junction-junction valves. Valves are
    plotted in the center between two junctions with a "helper" line (dashed and thin) being drawn
    between the junctions as well.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param valves: The valves for which the collections are created. If None, all valves which have\
        entries in the respective junction geodata will be plotted.
    :type valves: list, default None
    :param size: Patch size
    :type size: float, default 5.
    :param junction_geodata: GeoJSON strings to use for plotting. If None, net.junction.geo is used.
    :type junction_geodata: pandas.Series, default None
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param picker: Picker argument passed to the patch collection
    :type picker: bool, default False
    :param fill_closed: If True, valves with parameter opened == False will be filled and those\
        with opened == True will have a white facecolor. Vice versa if False.
    :type fill_closed: bool, default True
    :param kwargs: Keyword arguments are passed to the patch function
    :return: lc - line collection, pc - patch collection

    """
    valves = get_index_array(
        valves, net.valve[net.valve.opened.values].index if respect_valves else net.valve.index)

    valve_table = net.valve.loc[valves]

    geos, valves_with_geo = coords_from_node_geodata(
        element_indices=valves,
        from_nodes=valve_table.from_junction.values,
        to_nodes=valve_table.to_junction.values,
        node_geodata=junction_geodata if junction_geodata is not None else net.junction.geo,
        table_name="valve",
        node_name="Junction"
    )

    coords = [_get_coords_from_geojson(geo) for geo in geos]

    if len(valves_with_geo) == 0:
        return None

    colors = kwargs.pop("color", "k")
    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)
    patch_edgecolor = kwargs.pop("patch_edgecolor", colors)
    line_color = kwargs.pop("line_color", colors)

    infos = list(np.repeat([infofunc(i) for i in range(len(valves_with_geo))], 2)) \
        if infofunc is not None else []
    filled = valve_table["opened"].values
    if fill_closed:
        filled = ~filled
    pc, lc = _create_complex_branch_collection(
        coords, valve_patches, size, infos, picker=picker, linewidths=linewidths, filled=filled,
        patch_edgecolor=patch_edgecolor, line_color=line_color, **kwargs)

    return pc, lc


def create_flow_control_collection(net, flow_controllers=None, size=5., junction_geodata=None,
                                   infofunc=None, picker=False, fill_closed=True,
                                   respect_in_service=False, **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes flow control components.

    They are plotted in the center between two junctions and look like a valve with a T on top,
    if the flow control is active and an I on top, if the flow control is not active.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param flow_controllers: The flow_controllers for which the collections are created. If None,
        all flow_controllers which have entries in the respective junction geodata will be plotted.
    :type flow_controllers: list, default None
    :param size: Patch size
    :type size: float, default 5.
    :param junction_geodata: Coordinates to use for plotting. If None, net["junction_geodata"] is used.
    :type junction_geodata: pandas.DataFrame, default None
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param picker: Picker argument passed to the patch collection
    :type picker: bool, default False
    :param fill_closed: If True, flow_controllers with parameter in_service == False will be filled
        and those with in_service == True will have a white facecolor. Vice versa if False.
    :type fill_closed: bool, default True
    :param respect_in_service: if True, out-of-service flow controllers will not be plotted
    :type respect_in_service: bool default False
    :param kwargs: Keyword arguments are passed to the patch function
    :return: lc - line collection, pc - patch collection

    """
    flow_controllers = get_index_array(
        flow_controllers, net.flow_control[net.flow_control.in_service.values].index if
        respect_in_service else
        net.flow_control.index)

    fc_table = net.flow_control.loc[flow_controllers]

    geos, fc_with_geo = coords_from_node_geodata(
        flow_controllers, fc_table.from_junction.values, fc_table.to_junction.values,
        junction_geodata if junction_geodata is not None else net.junction.geo,
        "flow_control", "Junction")
    coords = [_get_coords_from_geojson(geo) for geo in geos]

    if len(fc_with_geo) == 0:
        return None

    colors = kwargs.pop("color", "k")
    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)
    patch_edgecolor = kwargs.pop("patch_edgecolor", colors)
    line_color = kwargs.pop("line_color", colors)

    infos = list(np.repeat([infofunc(i) for i in range(len(fc_with_geo))], 2)) \
        if infofunc is not None else []
    filled = fc_table["in_service"].values
    controlled = fc_table["control_active"].values
    if fill_closed:
        filled = ~filled
    pc, lc = _create_complex_branch_collection(
        coords, flow_control_patches, size, infos, picker=picker, linewidths=linewidths,
        filled=filled, patch_edgecolor=patch_edgecolor, line_color=line_color,
        controlled=controlled, **kwargs)

    return pc, lc


def create_pump_collection(net, pumps=None, table_name='pump', size=5., junction_geodata=None,
                           infofunc=None, picker=False, fj_col="from_junction",
                           tj_col="to_junction", **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes pumps.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param pumps: The pumps for which the collections are created. If None, all pumps which have\
        entries in the respective junction geodata will be plotted.
    :type pumps: list, default None
    :param table_name: Name of the pump table from which to get the data.
    :type table_name: str, default 'pump'
    :param size: Patch size
    :type size: float, default 5.
    :param junction_geodata: Coordinates to use for plotting. If None, net["junction_geodata"] is \
        used.
    :type junction_geodata: pandas.DataFrame, default None
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param fj_col: name of the from_junction column (can be different for different pump types)
    :type fj_col: str, default "from_junction"
    :param fj_col: name of the to_junction column (can be different for different pump types)
    :type fj_col: str, default "to_junction"
    :param picker: Picker argument passed to the patch collection
    :type picker: bool, default False
    :param kwargs: Keyword arguments are passed to the patch function
    :return: lc - line collection, pc - patch collection

    """
    pumps = get_index_array(pumps, net[table_name].index)
    pump_table = net[table_name].loc[pumps]

    geos, pumps_with_geo = coords_from_node_geodata(
        pumps, pump_table[fj_col].values, pump_table[tj_col].values,
        junction_geodata if junction_geodata is not None else net.junction.geo, "pump",
        "Junction")
    coords = [_get_coords_from_geojson(geo) for geo in geos]

    if len(pumps_with_geo) == 0:
        return None

    colors = kwargs.pop("color", "k")
    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)
    patch_edgecolor = kwargs.pop("patch_edgecolor", colors)
    line_color = kwargs.pop("line_color", colors)

    infos = list(np.repeat([infofunc(i) for i in range(len(pumps_with_geo))], 2)) \
        if infofunc is not None else []
    pc, lc = _create_complex_branch_collection(
        coords, pump_patches, size, infos, picker=picker, linewidths=linewidths,
        patch_edgecolor=patch_edgecolor, line_color=line_color, **kwargs)

    return pc, lc


def create_pressure_control_collection(net, pcs=None, table_name='press_control',
                                       size=5., junction_geodata=None,
                                       color='k', infofunc=None, picker=False, **kwargs):
    """Creates a matplotlib patch collection of pandapipes pressure controllers.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param pcs: The pressure controllers for which the collections are created. If None,
                all pressure controllers which have entries in the respective junction geodata
                will be plotted.
    :type pcs: list, default None
    :param size: Patch size
    :type size: float, default 5.
    :param junction_geodata: Coordinates to use for plotting. If None, net["junction_geodata"] is \
        used.
    :type junction_geodata: pandas.DataFrame, default None
    :param colors: Color or list of colors for every valve
    :type colors: iterable, float, default None
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param picker: Picker argument passed to the patch collection
    :type picker: bool, default False
    :param kwargs: Keyword arguments are passed to the patch function
    :return: lc - line collection, pc - patch collection

    """
    pcs = get_index_array(pcs, net[table_name].index)
    pc_table = net[table_name].loc[pcs]

    geos, pcs_with_geo = coords_from_node_geodata(
        pcs, pc_table.from_junction.values, pc_table.to_junction.values,
        junction_geodata if junction_geodata is not None else net.junction.geo, table_name,
        "Junction")
    coords = [_get_coords_from_geojson(geo) for geo in geos]

    if len(pcs_with_geo) == 0:
        return None

    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)

    infos = list(np.repeat([infofunc(i) for i in range(len(pcs_with_geo))], 2)) \
        if infofunc is not None else []
    pc, lc = _create_complex_branch_collection(coords, pressure_control_patches, size, infos,
                                               picker=picker, linewidths=linewidths,
                                               patch_edgecolor=color, line_color=color,
                                               **kwargs)

    return pc, lc


def create_compressor_collection(net, cmprs=None, table_name='compressor', size=5.,
                                 junction_geodata=None, color='k', infofunc=None, picker=False,
                                 **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes compressors. Compressors are
    plotted in the center between two junctions.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param cmprs: The compressors for which the collections are created. If None, all compressor
                 which have entries in the respective junction geodata will be plotted.
    :type cmprs: list, default None
    :param size: Patch size
    :type size: float, default 5.
    :param junction_geodata: Coordinates to use for plotting. If None, net["junction_geodata"] is \
        used.
    :type junction_geodata: pandas.DataFrame, default None
    :param colors: Color or list of colors for every compressor
    :type colors: iterable, float, default None
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param picker: Picker argument passed to the patch collection
    :type picker: bool, default False
    :param kwargs: Keyword arguments are passed to the patch function
    :return: lc - line collection, pc - patch collection

    """
    cmprs = get_index_array(cmprs, net[table_name].index)
    cmpr_table = net[table_name].loc[cmprs]

    coords, cmprs_with_geo = coords_from_node_geodata(
        cmprs, cmpr_table.from_junction.values, cmpr_table.to_junction.values,
        junction_geodata if junction_geodata is not None else net["junction_geodata"], table_name,
        "Junction")

    if len(cmprs_with_geo) == 0:
        return None

    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)

    infos = list(np.repeat([infofunc(i) for i in range(len(cmprs_with_geo))], 2)) \
        if infofunc is not None else []
    pc, lc = _create_complex_branch_collection(coords, compressor_patches, size, infos,
                                               picker=picker, linewidths=linewidths,
                                               patch_edgecolor=color, line_color=color,
                                               **kwargs)

    return pc, lc