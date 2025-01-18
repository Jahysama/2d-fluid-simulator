import taichi as ti

from advection import advect_kk_scheme, advect_upwind
from boundary_condition import get_boundary_condition
from solver import CipMacSolver, DyeCipMacSolver, DyeMacSolver, MacSolver
from pressure_updater import RedBlackSorPressureUpdater
from vorticity_confinement import VorticityConfinement
from visualization import visualize_norm, visualize_pressure, visualize_vorticity

@ti.data_oriented
class FluidSimulatorConfig:
    def __init__(self,
                 bg_color=(0.26, 0.88, 0.68),      # Background color
                 wall_color=(0.5, 0.0, 0.5),    # Wall color
                 dye_color=(0.79, 0.65, 0.4),    # Wall color
                 blend_factor=1.0,              # How much to blend with background
                 norm_factor=1.0,               # Velocity visualization intensity
                 pressure_factor=0.1,         # Pressure visualization intensity
                 vorticity_factor=0.00001):       # Vorticity visualization intensity
        self.bg_color = ti.Vector(list(bg_color))
        self.wall_color = ti.Vector(list(wall_color))
        self.dye_color = ti.Vector(list(dye_color))
        self.blend_factor = blend_factor
        self.norm_factor = norm_factor
        self.pressure_factor = pressure_factor
        self.vorticity_factor = vorticity_factor

@ti.data_oriented
class FluidSimulator:
    def __init__(self, solver, config=None):
        self._solver = solver
        self.rgb_buf = ti.Vector.field(3, ti.f32, shape=solver._resolution)
        
        # Use default config if none provided
        if config is None:
            config = FluidSimulatorConfig()
        self.config = config
        
        # Store colors and factors as Taichi vectors/fields
        self._bg_color = self.config.bg_color
        self._wall_color = self.config.wall_color
        self._dye_color = self.config.dye_color
        self._blend_factor = self.config.blend_factor
        self._norm_factor = self.config.norm_factor
        self._pressure_factor = self.config.pressure_factor
        self._vorticity_factor = self.config.vorticity_factor

    def step(self):
        self._solver.update()

    def update_colors(self, new_config):
        """Update visualization parameters at runtime"""
        self.config = new_config
        self._bg_color = self.config.bg_color
        self._wall_color = self.config.wall_color
        self._dye_color = self.config.dye_color
        self._blend_factor = self.config.blend_factor
        self._norm_factor = self.config.norm_factor
        self._pressure_factor = self.config.pressure_factor
        self._vorticity_factor = self.config.vorticity_factor

    @ti.kernel
    def _to_norm(self, rgb_buf: ti.template(), vc: ti.template(), pc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = self._bg_color
            if not self._solver.is_wall(i, j):
                rgb_buf[i, j] *= self._blend_factor
                rgb_buf[i, j] += self._norm_factor * visualize_norm(vc[i, j])
                rgb_buf[i, j] += self._pressure_factor * visualize_pressure(pc[i, j])
            else:
                rgb_buf[i, j] = self._wall_color

    @ti.kernel
    def _to_pressure(self, rgb_buf: ti.template(), pc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = self._bg_color
            if not self._solver.is_wall(i, j):
                rgb_buf[i, j] *= self._blend_factor
                rgb_buf[i, j] += self._pressure_factor * 20 * visualize_pressure(pc[i, j])
            else:
                rgb_buf[i, j] = self._wall_color

    @ti.kernel
    def _to_vorticity(self, rgb_buf: ti.template(), vc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = self._bg_color
            if not self._solver.is_wall(i, j):
                rgb_buf[i, j] *= self._blend_factor
                rgb_buf[i, j] += self._vorticity_factor * visualize_vorticity(vc, i, j, self._solver.dx)
            else:
                rgb_buf[i, j] = self._wall_color


@ti.data_oriented
class DyeFluidSimulator(FluidSimulator):
    def get_dye_field(self):
        self._to_dye(self.rgb_buf, self._solver.get_fields()[2])
        return self.rgb_buf

    @ti.kernel
    def _to_dye(self, rgb_buf: ti.template(), dye: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = self._bg_color
            if not self._solver.is_wall(i, j):
                rgb_buf[i, j] *= self._blend_factor
                # Multiply dye intensity by the dye color
                rgb_buf[i, j] += self._dye_color * dye[i, j][0]  # Use first channel for intensity
            else:
                rgb_buf[i, j] = self._wall_color

    def field_to_numpy(self):
        fields = self._solver.get_fields()
        return {"v": fields[0].to_numpy(), "p": fields[1].to_numpy(), "dye": fields[2].to_numpy()}

    @staticmethod
    def create(num, resolution, dt, dx, re, vor_eps, scheme):
        boundary_condition = get_boundary_condition(num, resolution, False)
        vorticity_confinement = (
            VorticityConfinement(boundary_condition, dt, dx, vor_eps)
            if vor_eps is not None
            else None
        )
        pressure_updater = RedBlackSorPressureUpdater(
            boundary_condition, dt, dx, relaxation_factor=1.3, n_iter=2
        )

        if scheme == "cip":
            solver = DyeCipMacSolver(
                boundary_condition, pressure_updater, dt, dx, re, vorticity_confinement
            )
        elif scheme == "upwind":
            solver = DyeMacSolver(
                boundary_condition,
                pressure_updater,
                advect_upwind,
                dt,
                dx,
                re,
                vorticity_confinement,
            )
        elif scheme == "kk":
            solver = DyeMacSolver(
                boundary_condition,
                pressure_updater,
                advect_kk_scheme,
                dt,
                dx,
                re,
                vorticity_confinement,
            )

        return DyeFluidSimulator(solver)
