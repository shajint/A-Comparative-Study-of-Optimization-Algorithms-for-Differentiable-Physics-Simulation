import numpy as onp
import jax
import jax.numpy as np
import jax.flatten_util
from dataclasses import dataclass
import functools
from typing import Any

from .utils import timeit
from .generate_mesh import Mesh
from .fe import FiniteElement
from . import logger

from .geometry import (
    FIXED_BOUNDS,
    compute_coil_areas,
    compute_fill_fractions,
    define_core_rectangles,
)

MU0 = 4 * np.pi * 1e-7
COIL_GRID_RES = 200


@dataclass
class Problem:
    """Problem class to handle one FE variable or multiple coupled FE variables.

    Attributes
    ----------
    mesh : Mesh
        :attr:`~new_fem_engine.fe.FiniteElement.mesh`
    vec : int
        :attr:`~new_fem_engine.fe.FiniteElement.vec`
    dim : int
        :attr:`~new_fem_engine.fe.FiniteElement.dim`
    ele_type : str
        :attr:`~new_fem_engine.fe.FiniteElement.ele_type`
    quadrature_rule
        :attr:`~new_fem_engine.fe.FiniteElement.quadrature_rule`
    quadrature_order : int
        :attr:`~new_fem_engine.fe.FiniteElement.quadrature_order`
    dirichlet_bc_info : list
        :attr:`~new_fem_engine.fe.FiniteElement.dirichlet_bc_info`
    location_fns : list
        A list of location functions useful for surface integrals in the weak form.
        Each callable takes a point (NumpyArray) and returns a boolean indicating
        if the point satisfies the location condition.
    additional_info : tuple
        Any other problem-dependent information.
    """
    mesh: Mesh
    vec: int
    dim: int
    ele_type: str = "QUAD4"
    quadrature_rule: Any = None
    quadrature_order: int = None
    dirichlet_bc_info: list = None
    location_fns: list = None
    additional_info: tuple = ()
    
    def __post_init__(self):
        # Wrap single-variable inputs into lists so coupled (multi-FE) problems
        # are handled uniformly. This project uses one variable (scalar A).
        if type(self.mesh) != type([]):
            self.mesh = [self.mesh]
            self.vec = [self.vec]
            self.ele_type = [self.ele_type]
            self.quadrature_rule = [self.quadrature_rule]
            self.quadrature_order = [self.quadrature_order]
            self.dirichlet_bc_info = [self.dirichlet_bc_info]

        self.num_vars = len(self.mesh)

        self.fes = [FiniteElement(mesh=self.mesh[i],
                                  vec=self.vec[i],
                                  dim=self.dim,
                                  ele_type=self.ele_type[i],
                                  quadrature_rule=self.quadrature_rule[i] if type(self.quadrature_rule) == type([]) else self.quadrature_rule,
                                  quadrature_order=self.quadrature_order[i] if type(self.quadrature_order) == type([]) else self.quadrature_order,
                                  dirichlet_bc_info=self.dirichlet_bc_info[i] if type(self.dirichlet_bc_info) == type([]) else self.dirichlet_bc_info)
                    for i in range(self.num_vars)]

        self.cells_list = [fe.cells for fe in self.fes]
        # Assume all fes have the same number of cells, same dimension
        self.num_cells = self.fes[0].num_cells
        self.boundary_inds_list = self.fes[0].get_boundary_conditions_inds(self.location_fns)

        self.offset = [0]
        for i in range(len(self.fes) - 1):
            self.offset.append(self.offset[i] + self.fes[i].num_total_dofs)
            
        def find_ind(*x):
            inds = []
            for i in range(len(x)):
                crt_ind = self.fes[i].vec * x[i][:, None] + np.arange(self.fes[i].vec)[None, :] + self.offset[i]
                inds.append(crt_ind.reshape(-1))
            return np.hstack(inds)

        # (num_cells, num_nodes*vec + ...)
        inds = onp.array(jax.vmap(find_ind)(*self.cells_list))
        self.I = onp.repeat(inds[:, :, None], inds.shape[1], axis=2).reshape(-1)
        self.J = onp.repeat(inds[:, None, :], inds.shape[1], axis=1).reshape(-1)
        self.cells_list_face_list = []

        for i, boundary_inds in enumerate(self.boundary_inds_list):
            cells_list_face = [cells[boundary_inds[:, 0]] for cells in self.cells_list]  # [(num_selected_faces, num_nodes), ...]
            inds_face = onp.array(jax.vmap(find_ind)(*cells_list_face))  # (num_selected_faces, num_nodes*vec + ...)
            I_face = onp.repeat(inds_face[:, :, None], inds_face.shape[1], axis=2).reshape(-1)
            J_face = onp.repeat(inds_face[:, None, :], inds_face.shape[1], axis=1).reshape(-1)
            self.I = onp.hstack((self.I, I_face))
            self.J = onp.hstack((self.J, J_face))
            self.cells_list_face_list.append(cells_list_face)

        self.cells_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*self.cells_list)  # (num_cells, num_nodes + ...)

        dumb_array_dof = [np.zeros((fe.num_nodes, fe.vec)) for fe in self.fes]
        dumb_array_node = [np.zeros(fe.num_nodes) for fe in self.fes]
        _, self.unflatten_fn_dof = jax.flatten_util.ravel_pytree(dumb_array_dof)

        dumb_sol_list = [np.zeros((fe.num_total_nodes, fe.vec)) for fe in self.fes]
        dumb_dofs, self.unflatten_fn_sol_list = jax.flatten_util.ravel_pytree(dumb_sol_list)
        self.num_total_dofs_all_vars = len(dumb_dofs)

        self.num_nodes_cumsum = onp.cumsum([0] + [fe.num_nodes for fe in self.fes])

        self.initialize_geometric_quantities()

        self.internal_vars = ()
        self.internal_vars_surfaces = [() for _ in range(len(self.boundary_inds_list))]
        self.custom_init(*self.additional_info)
        self.pre_jit_fns()
        
    def initialize_geometric_quantities(self, fes_points=None):
        """Initialize geometric quantities for the problem."""
        fes_points = [fe.points for fe in self.fes] if fes_points is None else fes_points

        for i, fe in enumerate(self.fes):
            # (num_cells, num_quads, num_nodes, dim), (num_cells, num_quads)
            fe.shape_grads, fe.JxW = fe.get_shape_grads(fes_points[i])
            # (num_cells, num_quads, num_nodes, 1, dim)
            fe.v_grads_JxW = fe.shape_grads[:, :, :, None, :] * fe.JxW[:, :, None, None, None]

        # Use jax.numpy so differentiable nodal coords (JAX arrays) are not cast to NumPy here.
        # (num_cells, num_vars, num_quads)
        self.JxW = np.transpose(np.stack([fe.JxW for fe in self.fes]), axes=(1, 0, 2))
        # (num_cells, num_quads, num_nodes + ..., dim)
        self.shape_grads = np.concatenate([fe.shape_grads for fe in self.fes], axis=2)
        # (num_cells, num_quads, num_nodes + ..., 1, dim)
        self.v_grads_JxW = np.concatenate([fe.v_grads_JxW for fe in self.fes], axis=2)

        # TODO: Now assumes all vars share the same quad points
        # (num_cells, num_quads, dim)
        self.physical_quad_points = self.fes[0].get_physical_quad_points(fes_points[0])

        self.selected_face_shape_grads = []
        self.nanson_scale = []
        self.selected_face_shape_vals = []
        self.physical_surface_quad_points = []
        for boundary_inds in self.boundary_inds_list:
            s_shape_grads = []
            n_scale = []
            s_shape_vals = []
            for i, fe in enumerate(self.fes):
                # (num_selected_faces, num_face_quads, num_nodes, dim), (num_selected_faces, num_face_quads)
                face_shape_grads_physical, nanson_scale = fe.get_face_shape_grads(boundary_inds, fes_points[i])
                selected_face_shape_vals = fe.face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
                s_shape_grads.append(face_shape_grads_physical)
                n_scale.append(nanson_scale)
                s_shape_vals.append(selected_face_shape_vals)

            # (num_selected_faces, num_face_quads, num_nodes + ..., dim)
            s_shape_grads = np.concatenate(s_shape_grads, axis=2)
            # (num_selected_faces, num_vars, num_face_quads)
            n_scale = np.transpose(np.stack(n_scale), axes=(1, 0, 2))
            # (num_selected_faces, num_face_quads, num_nodes + ...)
            s_shape_vals = np.concatenate(s_shape_vals, axis=2)
            # (num_selected_faces, num_face_quads, dim)
            physical_surface_quad_points = self.fes[0].get_physical_surface_quad_points(boundary_inds, fes_points[0])

            self.selected_face_shape_grads.append(s_shape_grads)
            self.nanson_scale.append(n_scale)
            self.selected_face_shape_vals.append(s_shape_vals)
            # TODO: Now assumes all vars share the same quad points
            self.physical_surface_quad_points.append(physical_surface_quad_points)
            
    def custom_init(self):
        """Child class should override if more things need to be done in initialization."""
        pass

    def get_laplace_kernel(self, tensor_map):
        """Generic stiffness kernel: ∫ tensor_map(∇u) · ∇v  JxW  dΩ.

        tensor_map encodes the physics (e.g., 1/μ, the 2πr axisymmetric
        weight, etc.) and is provided by the child class get_tensor_map.
        """

        def laplace_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_shape_grads = cell_shape_grads[:, :self.fes[0].num_nodes, :]
            cell_sol = cell_sol_list[0]
            cell_v_grads_JxW = cell_v_grads_JxW[:, :self.fes[0].num_nodes, :, :]
            vec = self.fes[0].vec

            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
            u_grads = np.sum(u_grads, axis=1)  # (num_quads, vec, dim)
            u_grads_reshape = u_grads.reshape(-1, vec, self.dim)  # (num_quads, vec, dim)
            # Only the *material* internal var (index 0) feeds tensor_map; extra
            # internal vars (e.g. traced coil state) are consumed by mass_map.
            # ``[:1]`` also keeps zero-internal-var problems (plain Laplace)
            # working, since ``*()`` passes no argument to a 1-arg tensor_map.
            # (num_quads, vec, dim)
            u_physics = jax.vmap(tensor_map)(u_grads_reshape, *cell_internal_vars[:1]).reshape(u_grads.shape)
            # (num_quads, num_nodes, vec, dim) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))
            val = jax.flatten_util.ravel_pytree(val)[0]  # (num_nodes*vec + ...,)
            return val

        return laplace_kernel
    
    def get_mass_kernel(self, mass_map):

        def mass_kernel(cell_sol_flat, x, cell_JxW, *cell_internal_vars):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim)
            # cell_JxW: (num_vars, num_quads)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]
            cell_JxW = cell_JxW[0]
            vec = self.fes[0].vec
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, num_nodes, vec) -> (num_quads, vec)
            u = np.sum(cell_sol[None, :, :] * self.fes[0].shape_vals[:, :, None], axis=1)
            u_physics = jax.vmap(mass_map)(u, x, *cell_internal_vars)  # (num_quads, vec)
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :] * self.fes[0].shape_vals[:, :, None] * cell_JxW[:, None, None], axis=0)
            val = jax.flatten_util.ravel_pytree(val)[0]  # (num_nodes*vec + ...,)
            return val

        return mass_kernel
    
    def get_surface_kernel(self, surface_map):

        def surface_kernel(cell_sol_flat, x, face_shape_vals, face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
            # face_shape_vals: (num_face_quads, num_nodes + ...)
            # face_shape_grads: (num_face_quads, num_nodes + ..., dim)
            # x: (num_face_quads, dim)
            # face_nanson_scale: (num_vars, num_face_quads)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]
            face_shape_vals = face_shape_vals[:, :self.fes[0].num_nodes]
            face_nanson_scale = face_nanson_scale[0]

            # (1, num_nodes, vec) * (num_face_quads, num_nodes, 1) -> (num_face_quads, vec)
            u = np.sum(cell_sol[None, :, :] * face_shape_vals[:, :, None], axis=1)
            u_physics = jax.vmap(surface_map)(u, x, *cell_internal_vars_surface)  # (num_face_quads, vec)
            # (num_face_quads, 1, vec) * (num_face_quads, num_nodes, 1) * (num_face_quads, 1, 1) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :] * face_shape_vals[:, :, None] * face_nanson_scale[:, None, None], axis=0)

            return jax.flatten_util.ravel_pytree(val)[0]

        return surface_kernel
    
    def pre_jit_fns(self):
        """Prepare the JIT-compiled assembly functions for this problem."""

        def value_and_jacfwd(f, x):
            pushfwd = functools.partial(jax.jvp, f, (x, ))
            basis = np.eye(len(x.reshape(-1)), dtype=x.dtype).reshape(-1, *x.shape)
            y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis, ))
            return y, jac

        def get_kernel_fn_cell():
            def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
                """Per-cell operator: builds the residual for one element.

                Three physics hooks, each optional (guarded by hasattr):
                - get_tensor_map()    -> stiffness ∫ tensor_map(∇u)·∇v JxW
                - get_mass_map()      -> load      ∫ mass_map(u, x)·v JxW
                - get_universal_kernel() -> custom operator covering cases the
                  two above cannot express (e.g., a dedicated axisymmetric
                  operator). Defined by the child problem when needed.
                """
                
                # TODO: If there is no kernel map, returning 0. is not a good choice. 
                # Return a zero array with proper shape will be better.
                if hasattr(self, 'get_mass_map'):
                    mass_kernel = self.get_mass_kernel(self.get_mass_map())
                    mass_val = mass_kernel(cell_sol_flat, physical_quad_points, cell_JxW, *cell_internal_vars)
                else:
                    mass_val = 0.

                if hasattr(self, 'get_tensor_map'):
                    laplace_kernel = self.get_laplace_kernel(self.get_tensor_map())
                    laplace_val = laplace_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars)
                else:
                    laplace_val = 0.

                if hasattr(self, 'get_universal_kernel'):
                    universal_kernel = self.get_universal_kernel()
                    universal_val = universal_kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, 
                        cell_v_grads_JxW, *cell_internal_vars)
                else:
                    universal_val = 0.

                return laplace_val + mass_val + universal_val
            
            def kernel_jac(cell_sol_flat, *args):
                kernel_partial = lambda cell_sol_flat: kernel(cell_sol_flat, *args)
                return value_and_jacfwd(kernel_partial, cell_sol_flat)  # kernel(cell_sol_flat, *args), jax.jacfwd(kernel)(cell_sol_flat, *args)
            
            return kernel, kernel_jac

        def get_kernel_fn_face(ind):
            """Face kernel + Jacobian for surface integrals (Neumann terms).

            Two optional hooks (guarded by hasattr):
            - get_surface_maps()                -> per-surface surface_map
            - get_universal_kernels_surface()   -> custom face operators
            Your MagnetostaticProblem defines neither initially, so all
            surface contributions are 0 and the face machinery sits dormant.
            """

            def kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals, face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
                
                if hasattr(self, 'get_surface_maps'):
                    surface_kernel = self.get_surface_kernel(self.get_surface_maps()[ind])
                    surface_val = surface_kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals,
                        face_shape_grads, face_nanson_scale, *cell_internal_vars_surface)
                else:
                    surface_val = 0.

                if hasattr(self, 'get_universal_kernels_surface'):
                    universal_kernel = self.get_universal_kernels_surface()[ind]
                    universal_val = universal_kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals,
                        face_shape_grads, face_nanson_scale, *cell_internal_vars_surface)
                else:
                    universal_val = 0.

                return surface_val + universal_val

            def kernel_jac(cell_sol_flat, *args):
                # return jax.jacfwd(kernel)(cell_sol_flat, *args)
                kernel_partial = lambda cell_sol_flat: kernel(cell_sol_flat, *args)
                return value_and_jacfwd(kernel_partial, cell_sol_flat)  # kernel(cell_sol_flat, *args), jax.jacfwd(kernel)(cell_sol_flat, *args)

            return kernel, kernel_jac

        kernel, kernel_jac = get_kernel_fn_cell()
        kernel = jax.jit(jax.vmap(kernel))
        kernel_jac = jax.jit(jax.vmap(kernel_jac))
        self.kernel = kernel
        self.kernel_jac = kernel_jac

        num_surfaces = len(self.boundary_inds_list)
        if hasattr(self, 'get_surface_maps'):
            assert num_surfaces == len(self.get_surface_maps())
        elif hasattr(self, 'get_universal_kernels_surface'):
            assert num_surfaces == len(self.get_universal_kernels_surface())
        else:
            assert num_surfaces == 0, "Missing definitions for surface integral"

        self.kernel_face = []
        self.kernel_jac_face = []
        for i in range(len(self.boundary_inds_list)):
            kernel_face, kernel_jac_face = get_kernel_fn_face(i)
            kernel_face = jax.jit(jax.vmap(kernel_face))
            kernel_jac_face = jax.jit(jax.vmap(kernel_jac_face))
            self.kernel_face.append(kernel_face)
            self.kernel_jac_face.append(kernel_jac_face)
            
    # @timeit
    def split_and_compute_cell(self, cells_sol_flat, np_version, jac_flag, internal_vars):
        """Volume integral in weak form, computed in batches to bound memory.

        Chunks the vmap over all cells into num_cuts batches so large meshes
        do not exhaust memory. Used for both the residual (jac_flag=False)
        and its Jacobian (jac_flag=True).
        """
        vmap_fn = self.kernel_jac if jac_flag else self.kernel
        num_cuts = 20
        if num_cuts > self.num_cells:
            num_cuts = self.num_cells
        batch_size = self.num_cells // num_cuts
        input_collection = [cells_sol_flat, self.physical_quad_points, self.shape_grads,
                            self.JxW, self.v_grads_JxW, *internal_vars]

        if jac_flag:
            values = []
            jacs = []
            for i in range(num_cuts):
                if i < num_cuts - 1:
                    input_col = jax.tree_util.tree_map(lambda x: x[i * batch_size:(i + 1) * batch_size], input_collection)
                else:
                    input_col = jax.tree_util.tree_map(lambda x: x[i * batch_size:], input_collection)

                val, jac = vmap_fn(*input_col)
                values.append(val)
                jacs.append(jac)
            values = np_version.vstack(values)
            jacs = np_version.vstack(jacs)

            return values, jacs
        else:
            values = []
            for i in range(num_cuts):
                if i < num_cuts - 1:
                    input_col = jax.tree_util.tree_map(lambda x: x[i * batch_size:(i + 1) * batch_size], input_collection)
                else:
                    input_col = jax.tree_util.tree_map(lambda x: x[i * batch_size:], input_collection)

                val = vmap_fn(*input_col)
                values.append(val)
            values = np_version.vstack(values)
            return values
        
    def compute_face(self, cells_sol_flat, np_version, jac_flag, internal_vars_surfaces):
        """Surface integral in weak form
        """
        if jac_flag:
            values = []
            jacs = []
            for i, boundary_inds in enumerate(self.boundary_inds_list):
                vmap_fn = self.kernel_jac_face[i]
                selected_cell_sols_flat = cells_sol_flat[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes*vec + ...))
                input_collection = [selected_cell_sols_flat, self.physical_surface_quad_points[i], self.selected_face_shape_vals[i], 
                                    self.selected_face_shape_grads[i], self.nanson_scale[i], *internal_vars_surfaces[i]]

                val, jac = vmap_fn(*input_collection)
                values.append(val)
                jacs.append(jac)
            return values, jacs
        else:
            values = []
            for i, boundary_inds in enumerate(self.boundary_inds_list):
                vmap_fn = self.kernel_face[i]
                selected_cell_sols_flat = cells_sol_flat[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes*vec + ...))
                # TODO: duplicated code
                input_collection = [selected_cell_sols_flat, self.physical_surface_quad_points[i], self.selected_face_shape_vals[i], 
                                    self.selected_face_shape_grads[i], self.nanson_scale[i], *internal_vars_surfaces[i]]
                val = vmap_fn(*input_collection)
                values.append(val)
            return values

    def compute_residual_vars_helper(self, weak_form_flat, weak_form_face_flat):
        res_list = [np.zeros((fe.num_total_nodes, fe.vec), dtype=weak_form_flat.dtype) for fe in self.fes]
        weak_form_list = jax.vmap(lambda x: self.unflatten_fn_dof(x))(weak_form_flat) # [(num_cells, num_nodes, vec), ...]
        res_list = [res_list[i].at[self.cells_list[i].reshape(-1)].add(weak_form_list[i].reshape(-1, 
            self.fes[i].vec)) for i in range(self.num_vars)]

        for ind, cells_list_face in enumerate(self.cells_list_face_list):
            weak_form_face_list = jax.vmap(lambda x: self.unflatten_fn_dof(x))(weak_form_face_flat[ind]) # [(num_selected_faces, num_nodes, vec), ...]
            res_list = [res_list[i].at[cells_list_face[i].reshape(-1)].add(weak_form_face_list[i].reshape(-1, 
                self.fes[i].vec)) for i in range(self.num_vars)]   

        return res_list

    def compute_residual_vars(self, sol_list, internal_vars, internal_vars_surfaces):
        logger.debug(f"Computing cell residual...")
        cells_sol_list = [sol[cells] for cells, sol in zip(self.cells_list, sol_list)] # [(num_cells, num_nodes, vec), ...]
        cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list) # (num_cells, num_nodes*vec + ...)
        weak_form_flat = self.split_and_compute_cell(cells_sol_flat, np, False, internal_vars)  # (num_cells, num_nodes*vec + ...)
        weak_form_face_flat = self.compute_face(cells_sol_flat, np, False, internal_vars_surfaces)  # [(num_selected_faces, num_nodes*vec + ...), ...]
        return self.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat)

    def compute_newton_vars(self, sol_list, internal_vars, internal_vars_surfaces):
        logger.debug(f"Computing cell Jacobian and cell residual...")
        cells_sol_list = [sol[cells] for cells, sol in zip(self.cells_list, sol_list)] # [(num_cells, num_nodes, vec), ...]
        cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list) # (num_cells, num_nodes*vec + ...)
        # (num_cells, num_nodes*vec + ...),  (num_cells, num_nodes*vec + ..., num_nodes*vec + ...)
        weak_form_flat, cells_jac_flat = self.split_and_compute_cell(cells_sol_flat, np, True, internal_vars)
        V = cells_jac_flat.reshape(-1)

        # [(num_selected_faces, num_nodes*vec + ...,), ...], [(num_selected_faces, num_nodes*vec + ..., num_nodes*vec + ...,), ...]
        weak_form_face_flat, cells_jac_face_flat = self.compute_face(cells_sol_flat, np, True, internal_vars_surfaces)
        for cells_jac_f_flat in cells_jac_face_flat:
            V = np.hstack((V, cells_jac_f_flat.reshape(-1)))

        # Kept for backward compatibility: existing tests still read ``prob.V`` as
        # the ``get_A`` data source. Engine callers use the returned ``V`` instead.
        self.V = np.asarray(V)

        return self.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat), self.V

    def compute_residual(self, sol_list):
        """Given FE solution list, compute the residual list.

        Parameters
        ----------
        sol_list : list
            A list of JaxArray with the shape being (num_total_nodes, vec).

        Returns
        -------
        res_list : list
            Same shape as sol_list.
        """
        return self.compute_residual_vars(sol_list, self.internal_vars, self.internal_vars_surfaces)

    def newton_update(self, sol_list):
        """Given FE solution list, compute the tangent stiffness matrix, as well as the residual list.

        Parameters
        ----------
        sol_list : list
            A list of JaxArray with the shape being (num_total_nodes, vec).

        Returns
        -------
        res_list : list
            Same shape as sol_list.
        V : JaxArray
            Flat per-cell Jacobian entries (length matches ``self.I``/``self.J``),
            the ``get_A`` data source. Also mirrored on ``self.V``.
        """
        return self.compute_newton_vars(sol_list, self.internal_vars, self.internal_vars_surfaces)

    def set_params(self, params):
        """This is the key method for solving differentiable inverse problems.
        We MUST define (override) this method so that ``params`` become
        differentiable. No need to define this method if only the forward
        problem is solved.

        For the PQ-core geometry optimization, ``params`` is the fill-fraction
        field of magnetic material, defined on the element quadrature points::

            def set_params(self, params):
                # params: JaxPytree, shape (num_cells, num_quads)
                self.internal_vars = [params]

        The coupling to ``get_tensor_map`` is the essential part. A ``params``
        array with shape ``(num_cells, num_quads, shape1, shape2)`` is sliced by
        the JIT'd kernel into per-quadrature-point slices of shape
        ``(shape1, shape2)``, and ``get_tensor_map`` must consume them::

            class MagnetostaticProblem(Problem):
                def get_tensor_map(self):
                    def tensor_fn(u_grad, param):
                        # param MUST have shape (shape1, shape2)
                        # e.g. param holds 1/(mu0 * mu_rel) at this quad point
                        return ...
                    return tensor_fn

        For this project ``params`` has shape ``(num_cells, num_quads)`` with
        ``num_quads = 4`` for QUAD4, so the per-quadrature-point slice ``param``
        inside ``tensor_fn`` is a scalar.

        Once ``set_params`` is defined, ``fwd_pred`` becomes differentiable
        through the automatic differentiation wrapper::

            fwd_pred = ad_wrapper(problem)
            sol_list = fwd_pred(params)

        params: `JaxPytree <https://docs.jax.dev/en/latest/pytrees.html>`_
            The parameters to be differentiated.
        """
        raise NotImplementedError("Child class must implement this function!")

    def print_BC_info(self):
        """Print boundary and surface-integral set information for debugging."""
        boundary_inds_list = self.boundary_inds_list
        if len(boundary_inds_list) != 0:
            print("\n\n### Surface integral boundary sets are specified")
            for i in range(len(boundary_inds_list)):
                print(f"\nSurface boundary set {i + 1} information:")
                print(boundary_inds_list[i])
                print(
                    f"Array.shape = (num_selected_faces, 2) = {boundary_inds_list[i].shape}"
                )
                print("Interpretation:")
                print(
                    "    Array[i, 0] returns the global cell index of the ith selected face"
                )
                print(
                    "    Array[i, 1] returns the local face index of the ith selected face"
                )
        else:
            print("\n\n### No surface integral boundary sets found.")

        # Single finite element variable (A, scalar) in this project
        fe = self.fes[0]
        if len(fe.node_inds_list) != 0:
            print("\n\n### Dirichlet B.C. is specified")
            for i in range(len(fe.node_inds_list)):
                print(f"\nDirichlet Boundary part {i + 1} information:")
                if len(fe.node_inds_list[i]) == 0:
                    bc_array = onp.zeros((0, 3))
                else:
                    bc_array = onp.stack([
                        fe.node_inds_list[i],
                        fe.vec_inds_list[i],
                        fe.vals_list[i],
                    ]).T
                print(bc_array)
                print(
                    f"Array.shape = (num_selected_dofs, 3) = {bc_array.shape}"
                )
                print("Interpretation:")
                print(
                    "    Array[i, 0] returns the node index of the ith selected dof"
                )
                print(
                    "    Array[i, 1] returns the vec index of the ith selected dof"
                )
                print(
                    "    Array[i, 2] returns the value assigned to ith selected dof"
                )
        else:
            print("\n\n### No Dirichlet B.C. found.")


class MagnetostaticProblem(Problem):
    """2D axisymmetric magnetostatics for the PQ-core topology optimizer.

    State variable: psi = r * A_phi (radius times azimuthal magnetic vector
    potential).  The axisymmetric weak form is

        ∫ ν (2π / r) ∇ψ · ∇v  dr dz  =  ∫ 2π J v  dr dz,

    where ν = 1/μ is the (fill-fraction-averaged) magnetic reluctance and J is
    the azimuthal coil current density.  The 2π cancels on the source side
    because dV = 2πr dr dz and A_phi = ψ/r.  The stiffness (left) side is
    assembled by the Laplace kernel via ``get_tensor_map``; the source (right)
    side by the mass kernel via ``get_mass_map``.

    ``additional_info`` must be ``(core_spec, mesh_size)``, where ``core_spec``
    contains a ``"params"`` entry for ``new_fem_engine.geometry.define_core_rectangles``.
    """

    def custom_init(self, core_spec, mesh_size):
        self.mu0 = MU0
        self.mur = core_spec["params"]["mur"]
        self.mesh_size = mesh_size

        params = dict(core_spec["params"])
        params.setdefault("gap_number", 0)
        params.setdefault("gap_size", 0.0)

        (self.core_rects_padded, self.rect_mask,
         self.primary_rect, self.secondary_rect) = define_core_rectangles(params)

        rmin, rmax, zmin, zmax = FIXED_BOUNDS
        xs = np.linspace(rmin, rmax, COIL_GRID_RES)
        ys = np.linspace(zmin, zmax, COIL_GRID_RES)
        _, f_prim, f_sec, _ = compute_fill_fractions(
            xs, ys, self.core_rects_padded, self.rect_mask,
            self.primary_rect, self.secondary_rect, self.mur, MU0,
        )
        cell_area = (xs[1] - xs[0]) * (ys[1] - ys[0])
        self.area_prim, self.area_sec = compute_coil_areas(f_prim, f_sec, cell_area)

        # Primary and secondary coil sections are half-turns of the same single
        # winding (0.5 + 0.5 = 1 full turn), matching the reference FE solver.
        self.turns = 0.5
        self.current = 1.0

    @staticmethod
    def _current_density(r, z, prim, sec, a_prim, a_sec, turns, current):
        """Pure coil-current density (A/m^2) — coil state passed explicitly.

        Keeping the coil geometry as plain traced arguments (not ``self``
        attributes) lets the mass kernel and inductance consume it under
        ``jax.jit``, so the optimizer loop can move the coils without
        recompiling the fused forward+backward.
        """
        r0, z0, r1, z1 = prim
        in_primary = (r >= r0) & (r <= r1) & (z >= z0) & (z <= z1)
        J_prim = (turns * current / a_prim) * in_primary

        r0, z0, r1, z1 = sec
        in_secondary = (r >= r0) & (r <= r1) & (z >= z0) & (z <= z1)
        J_sec = np.where(
            a_sec > 0,
            (turns * current / a_sec) * in_secondary,
            0.0,
        )
        return J_prim + J_sec

    def _current_density_at_point(self, r, z):
        """Eager convenience: coil density from the current ``self`` state."""
        return self._current_density(
            r, z, self.primary_rect, self.secondary_rect,
            self.area_prim, self.area_sec, self.turns, self.current)

    def get_tensor_map(self):
        def tensor_fn(u_grad, nu_2pi_over_r):
            return nu_2pi_over_r * u_grad

        return tensor_fn

    def get_mass_map(self):
        # The engine assembles residual = laplace + mass and solves for 0, so the
        # load enters with a negative sign. The 2π matches the stiffness factor:
        # ∫J·δA·dV = ∫J(δψ/r)(2πr dr dz) = ∫2π·J·δψ dr dz (the r cancels).
        # ``*cell_internal_vars`` carries the traced coil state (see set_params).
        def mass_fn(u, x, nu_2pi_over_r, prim, sec, a_prim, a_sec):
            J = self._current_density(x[..., 0], x[..., 1], prim, sec,
                                      a_prim, a_sec, self.turns, self.current)
            return -2.0 * np.pi * J[..., None]

        return mass_fn

    def set_params(self, params):
        """Set the differentiable material fill and coil state.

        Parameters
        ----------
        params : JaxArray or tuple
            Either the fill fraction field ``(num_cells, num_quads)`` alone
            (backward compatible: coil geometry is read from ``self``, the
            certified/frozen-coil path), or a tuple
            ``(fill, primary_rect, secondary_rect, area_prim, area_sec)`` with
            traced coil geometry so the optimizer loop may move the coils
            without recompiling a jitted forward+backward.

        The ``nu`` field is stored as internal var index 0 (consumed by
        ``get_tensor_map``); the coil state is broadcast to per-quadrature
        shape and stored in indices 1..4 (consumed by ``get_mass_map`` and
        ``compute_inductance``).
        """
        if isinstance(params, tuple):
            fill, prim, sec, a_prim, a_sec = params
        else:
            fill = params
            prim, sec, a_prim, a_sec = (self.primary_rect, self.secondary_rect,
                                        self.area_prim, self.area_sec)
        mu_eff = 1.0 / (fill / self.mur + (1.0 - fill))
        nu = 1.0 / (MU0 * mu_eff)
        r = self.physical_quad_points[..., 0]
        r_safe = np.maximum(r, 1e-6)

        n_cells, n_quads = fill.shape
        prim_b = np.broadcast_to(prim, (n_cells, n_quads, 4))
        sec_b = np.broadcast_to(sec, (n_cells, n_quads, 4))
        a_prim_b = np.broadcast_to(a_prim, (n_cells, n_quads))
        a_sec_b = np.broadcast_to(a_sec, (n_cells, n_quads))
        self.internal_vars = [nu * (2 * np.pi / r_safe), prim_b, sec_b, a_prim_b, a_sec_b]

    def compute_inductance(self, sol_list, prim=None, sec=None, a_prim=None, a_sec=None):
        """Inductance from stored magnetic energy, L = 2π/I² ∫ J ψ dA.

        ``sol_list`` is the solved state.  ``prim``/``sec``/``a_prim``/``a_sec``
        are the coil geometry (passed explicitly so the traced/jitted path can
        feed its own tracers from the caller's trace scope — reading them from
        ``self.internal_vars`` instead would leak custom_vjp forward tracers
        across the transformation boundary).  When omitted, the eager path
        reads the ``self`` coil state (backward compatible).
        """
        if prim is None:
            prim, sec, a_prim, a_sec = (self.primary_rect, self.secondary_rect,
                                        self.area_prim, self.area_sec)
        psi = sol_list[0]
        psi_quads = self.fes[0].convert_from_dof_to_quad(psi)  # (num_cells, num_quads, vec)
        JxW = self.fes[0].JxW[:, :, None]  # (num_cells, num_quads, 1)
        pts = self.physical_quad_points  # (num_cells, num_quads, dim)
        # turns/current are fixed scalars; bind them (vmap can't map rank-0 args).
        n_cells, n_quads = psi_quads.shape[:2]
        prim_b = np.broadcast_to(prim, (n_cells, n_quads, 4))
        sec_b = np.broadcast_to(sec, (n_cells, n_quads, 4))
        a_prim_b = np.broadcast_to(a_prim, (n_cells, n_quads))
        a_sec_b = np.broadcast_to(a_sec, (n_cells, n_quads))
        dens = lambda r, z, p, s, ap, a_sec_1: self._current_density(
            r, z, p, s, ap, a_sec_1, self.turns, self.current)
        J_quads = jax.vmap(jax.vmap(dens))(
            pts[..., 0], pts[..., 1], prim_b, sec_b, a_prim_b, a_sec_b)
        J_quads = J_quads[:, :, None]  # (num_cells, num_quads, 1)
        energy = 2 * np.pi * np.sum(psi_quads * J_quads * JxW)
        return energy / self.current ** 2