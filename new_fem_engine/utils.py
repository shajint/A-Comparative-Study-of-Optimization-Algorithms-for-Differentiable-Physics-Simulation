import jax
import numpy as onp
import meshio
import os
import time
from functools import wraps

from . import logger
from .generate_mesh import get_meshio_cell_type

def save_sol(fe, sol, sol_file, cell_infos=None, point_infos=None):
    """
    Save finite element solution and associated data to VTK file.

    For the magnetostatics project, this is the visualization entry point:
    save the magnetic vector potential ``A`` (vertex data), and optionally
    per-cell fields (e.g., fill-fraction, |B|) and per-vertex fields
    (e.g., B vector components).

    Parameters
    ----------
    fe : FiniteElement
        Finite element object. Its ``points`` are the (r, z) coordinates of
        the 2D axisymmetric mesh (num_total_nodes, 2).
    sol : JaxArray
        Solution vector to save (vertex-based).
        Shape is (num_total_nodes, vec). For scalar A, vec = 1.
    sol_file : str
        Output file path.
    cell_infos : list
        Additional cell data as [(name1, data1), (name2, data2)].
        Each data array must have shape (num_cells,...).
        For example, ::

            cell_infos = [('fill_fraction', theta_cell_data)]

    point_infos : list
        Additional point data as [(name1, data1), (name2, data2)].
        Each data array must have shape (num_total_nodes,...).

        For example, ::

            point_infos = [('B', B_point_data)]
    """
    cell_type = get_meshio_cell_type(fe.ele_type)
    sol_dir = os.path.dirname(sol_file)
    os.makedirs(sol_dir, exist_ok=True)
    out_mesh = meshio.Mesh(points=fe.points, cells={cell_type: fe.cells})
    out_mesh.point_data['sol'] = onp.array(sol, dtype=onp.float32)
    if cell_infos is not None:
        for cell_info in cell_infos:
            name, data = cell_info
            # TODO: vector-valued cell data
            assert data.shape == (fe.num_cells,), f"cell data wrong shape, get {data.shape}, while num_cells = {fe.num_cells}"
            out_mesh.cell_data[name] = [onp.array(data, dtype=onp.float32)]
    if point_infos is not None:
        for point_info in point_infos:
            name, data = point_info
            assert len(data) == len(sol), "point data wrong shape!"
            out_mesh.point_data[name] = onp.array(data, dtype=onp.float32)
    out_mesh.write(sol_file)


def modify_vtu_file(input_file_path, output_file_path):
    """Convert VTK file version from 2.2 to 1.0 for compatibility.

    Notes
    -----
    meshio does not accept version 2.2, raising error of
    `meshio._exceptions.ReadError: Unknown VTU file version '2.2'.`
    Only relevant if you open externally-produced VTU files (e.g., from
    other solvers). Not needed for files written by ``save_sol``.

    Parameters
    ----------
    input_file_path : str
        Path to input VTU file (version 2.2)
    output_file_path : str
        Path for output VTU file (version 1.0)
    """
    fin = open(input_file_path, "r")
    fout = open(output_file_path, "w")
    for line in fin:
        fout.write(line.replace('<VTKFile type="UnstructuredGrid" version="2.2">', '<VTKFile type="UnstructuredGrid" version="1.0">'))
    fin.close()
    fout.close()
    
def timeit(func):
    """Decorator for printing the timing results of a function.

    Parameters
    ----------
    func : callable
        Function to be timed.

    Returns
    -------
    callable
        Wrapped function with timing logic.
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.debug(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def walltime(txt_dir=None, filename=None):
    """Wrapper for writing timing results to a file.

    Used for the lineax vs feax solver benchmark: annotate a function with
    this decorator and (optionally) point it at a directory; each run
    appends a fresh ``{filename}_{platform}.txt`` with
    ``start_time, end_time, time_elapsed``.

    Parameters
    ----------
    txt_dir : str
        Directory to save timing data.
    filename : str
        Base filename (default: 'walltime_{platform}.txt').

    Returns
    -------
    callable
        Decorator function.
    """

    def decorate(func):

        def wrapper(*list_args, **keyword_args):
            start_time = time.time()
            return_values = func(*list_args, **keyword_args)
            end_time = time.time()
            time_elapsed = end_time - start_time
            platform = jax.lib.xla_bridge.get_backend().platform
            logger.info(
                f"Time elapsed {time_elapsed} of function {func.__name__} "
                f"on platform {platform}"
            )
            if txt_dir is not None:
                os.makedirs(txt_dir, exist_ok=True)
                fname = 'walltime'
                if filename is not None:
                    fname = filename
                with open(os.path.join(txt_dir, f"{fname}_{platform}.txt"),
                          'w') as f:
                    f.write(f'{start_time}, {end_time}, {time_elapsed}\n')
            return return_values

        return wrapper

    return decorate