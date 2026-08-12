import jax
import jax.numpy as jnp

MAX_CORE_RECTS = 60
FIXED_BOUNDS = (0.0, 0.035, -0.03, 0.03)


def define_core_rectangles(params):
    r_center_post = params["center_post_diameter"] / 2
    r_inner = params["leg_inner_diameter"] / 2
    leg_height = params["leg_height"] / 2
    window_height = params["window_height"]
    gap_number = int(params["gap_number"])
    coil_clearance = params["coil_clearance"]

    yoke_thickness = leg_height - window_height / 2.0

    A_e = params.get("A_e", jnp.pi * r_center_post ** 2)
    phi = params.get("phi", 1.0)
    r_outer = jnp.sqrt(r_inner ** 2 + A_e / (jnp.pi * phi))

    rects = []

    def add_rect(r0, z0, r1, z1):
        rects.append(jnp.array([r0, z0, r1, z1]))

    yoke_taper = bool(params.get("yoke_taper", False))

    # Yoke over the center post (r = 0 .. r_center_post): full yoke_thickness.
    add_rect(0.0, leg_height - yoke_thickness, r_center_post, leg_height)
    add_rect(0.0, -leg_height, r_center_post, -leg_height + yoke_thickness)

    # Flux-conserving hyperbolic taper (real PQ yoke), mirroring the reference
    # FDM solver: h(r) = yoke_thickness * r_center_post / r keeps the annular
    # cross-section 2*pi*r*h(r) constant.  Flat yoke = h(r) == yoke_thickness.
    N_strips = 20
    r_strips = jnp.linspace(r_center_post, r_outer, N_strips + 1)
    for i in range(N_strips):
        r0 = r_strips[i]
        r1 = r_strips[i + 1]
        if yoke_taper:
            r_mid = 0.5 * (r0 + r1)
            h = yoke_thickness * r_center_post / r_mid
        else:
            h = yoke_thickness
        add_rect(r0, leg_height - yoke_thickness, r1, leg_height - yoke_thickness + h)
        add_rect(r0, -leg_height + yoke_thickness - h, r1, -leg_height + yoke_thickness)

    add_rect(r_inner, -window_height / 2.0, r_outer, window_height / 2.0)

    center_z_bottom = -window_height / 2.0
    center_z_top = window_height / 2.0

    use_gap = params.get("gap_shifts", None) is not None
    gap_shifts = params.get("gap_shifts", jnp.array([]))
    gap_widths = params.get("gap_widths", jnp.array([]))

    if use_gap:
        gap_number = len(gap_shifts)
        r1_solid = jnp.where(gap_number > 0, 0.0, r_center_post)
        add_rect(0.0, center_z_bottom, r1_solid, center_z_top)

        zone_size = (center_z_top - center_z_bottom) / gap_number
        z_start = center_z_bottom

        for k in range(gap_number):
            base_center = center_z_bottom + (k + 0.5) * zone_size
            wk = gap_widths[k]
            max_shift = jnp.maximum((zone_size - wk) / 2.0, 0.0)
            center = base_center + gap_shifts[k] * max_shift

            g0 = center - wk / 2.0
            g1 = center + wk / 2.0

            gap_active = g0 > z_start
            z0_seg = jnp.where(gap_active, z_start, 0.0)
            z1_seg = jnp.where(gap_active, g0, 0.0)
            r1_seg = jnp.where(gap_active, r_center_post, 0.0)
            add_rect(0.0, z0_seg, r1_seg, z1_seg)
            z_start = g1

        last_active = z_start < center_z_top
        z0_last = jnp.where(last_active, z_start, 0.0)
        z1_last = jnp.where(last_active, center_z_top, 0.0)
        r1_last = jnp.where(last_active, r_center_post, 0.0)
        add_rect(0.0, z0_last, r1_last, z1_last)
    else:
        gap_size = params.get("gap_size", 0.0)
        use_gap_size = gap_size > 0

        gap_num_eff = jnp.maximum(jnp.array(gap_number), 1)
        gap_each = gap_size / gap_num_eff
        spacing = (center_z_top - center_z_bottom) / gap_num_eff

        r1_solid = jnp.where(use_gap_size, 0.0, r_center_post)
        add_rect(0.0, center_z_bottom, r1_solid, center_z_top)

        z_start = center_z_bottom
        for k in range(gap_number):
            center = center_z_bottom + (k + 0.5) * spacing
            g0 = center - gap_each / 2.0
            g1 = center + gap_each / 2.0
            gap_active = use_gap_size & (g0 > z_start)
            z0_seg = jnp.where(gap_active, z_start, 0.0)
            z1_seg = jnp.where(gap_active, g0, 0.0)
            r1_seg = jnp.where(gap_active, r_center_post, 0.0)
            add_rect(0.0, z0_seg, r1_seg, z1_seg)
            z_start = g1

        last_active = use_gap_size & (z_start < center_z_top)
        z0_last = jnp.where(last_active, z_start, 0.0)
        z1_last = jnp.where(last_active, center_z_top, 0.0)
        r1_last = jnp.where(last_active, r_center_post, 0.0)
        add_rect(0.0, z0_last, r1_last, z1_last)

    r_coil_inner = r_center_post + coil_clearance
    r_coil_outer = r_inner - coil_clearance
    r_coil_mid = (r_coil_inner + r_coil_outer) / 2
    z_coil_bottom = -window_height / 2.0 + coil_clearance
    z_coil_top = window_height / 2.0 - coil_clearance

    # Radial ("layer") split with an air-insulation gap of ``coil_clearance``
    # between the two windings, mirroring the reference FDM solver's
    # ``coil_split='layer'`` (each coil pulled back coil_clearance/2 from mid).
    primary_rect = jnp.array([r_coil_inner, z_coil_bottom,
                              r_coil_mid - coil_clearance / 2.0, z_coil_top])
    secondary_rect = jnp.array([r_coil_mid + coil_clearance / 2.0,
                                z_coil_bottom, r_coil_outer, z_coil_top])

    n = len(rects)
    padded = jnp.zeros((MAX_CORE_RECTS, 4))
    for i in range(min(n, MAX_CORE_RECTS)):
        padded = padded.at[i].set(rects[i])
    mask = jnp.zeros(MAX_CORE_RECTS, dtype=jnp.bool_)
    mask = mask.at[:n].set(True)

    return padded, mask, primary_rect, secondary_rect


def intersect_area(cxmin, cxmax, cymin, cymax, rxmin, rymin, rxmax, rymax):
    ixmin = jnp.maximum(cxmin, rxmin)
    ixmax = jnp.minimum(cxmax, rxmax)
    iymin = jnp.maximum(cymin, rymin)
    iymax = jnp.minimum(cymax, rymax)
    return jnp.maximum(0.0, ixmax - ixmin) * jnp.maximum(0.0, iymax - iymin)


def compute_fill_fractions(xs, ys, core_rects_padded, rect_mask, primary_rect, secondary_rect, mur, mu0=None):
    if mu0 is None:
        mu0 = 4 * jnp.pi * 1e-7
    h = xs[1] - xs[0]
    nx = xs.shape[0]
    ny = ys.shape[0]

    xs_2d = jnp.broadcast_to(xs[None, :], (ny, nx))
    ys_2d = jnp.broadcast_to(ys[:, None], (ny, nx))

    cxmin = jnp.maximum(0.0, xs_2d - h / 2)
    cxmax = xs_2d + h / 2
    cymin = ys_2d - h / 2
    cymax = ys_2d + h / 2
    cell_area = (cxmax - cxmin) * (cymax - cymin)

    def rect_fill(rect):
        r0, z0, r1, z1 = rect
        return intersect_area(cxmin, cxmax, cymin, cymax, r0, z0, r1, z1) / cell_area

    f_core = jnp.zeros((ny, nx))
    for i in range(MAX_CORE_RECTS):
        f_core = f_core + jnp.where(rect_mask[i], rect_fill(core_rects_padded[i]), 0.0)

    f_core = jnp.clip(f_core, 0.0, 1.0)
    f_prim = jnp.clip(rect_fill(primary_rect), 0.0, 1.0)
    f_sec = jnp.clip(rect_fill(secondary_rect), 0.0, 1.0)

    mr_eff = jnp.where(
        f_core > 0.999,
        mur,
        jnp.where(f_core < 0.001, 1.0, 1.0 / (f_core / mur + (1.0 - f_core)))
    )
    k_arr = 1.0 / (mu0 * mr_eff)

    return k_arr, f_prim, f_sec, f_core


def compute_current_density(f_prim_map, f_sec_map, area_prim, area_sec, current=1.0, turns=1):
    J_prim = (turns * current / area_prim) * f_prim_map
    J_sec = jnp.where(area_sec > 0, (turns * current / area_sec) * f_sec_map, 0.0)
    return J_prim + J_sec


def compute_coil_areas(f_prim_map, f_sec_map, cell_area):
    return jnp.sum(f_prim_map * cell_area), jnp.sum(f_sec_map * cell_area)
