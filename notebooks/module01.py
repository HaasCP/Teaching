import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrow
from PIL import Image


def signif(x, p):
    """Round to p significant figures.

    Args:
        x (_type_): The number to round
        p (_type_): The number of significant figures

    Returns:
        _type_: The rounded number
    """
    # from https://github.com/pola-rs/polars/issues/11968
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def van_deemter_plot(A=5.0, B=1.0, C=5.0, u_min=0.05, u_max=3.0, show_terms=True):
    """
    Plot H(u) = A + B/u + C*u and indicate dominance regions for A, B/u, and C*u.
    Parameters are in arbitrary units; u is linear velocity.
    Units are pedagogical; axes labeled in mm and mm s^-1 for concreteness.
    """
    # guard rails
    u_min = max(1e-3, float(u_min))
    u_max = max(u_min + 1e-3, float(u_max))
    A = float(A)
    B = float(B)
    C = float(C)

    # domain
    n = 900
    u = np.linspace(u_min, u_max, n)

    # components and total
    A_arr = A * np.ones_like(u)
    # avoid division warnings by ensuring u_min > 0 (handled above)
    B_arr = B / u
    C_arr = C * u
    H = A_arr + B_arr + C_arr

    # optimum (only meaningful for B>=0 and C>0)
    u_opt = np.sqrt(B / C) if (B > 0 and C > 0) else np.nan
    H_min = A + 2.0 * np.sqrt(B * C) if (B >= 0 and C >= 0) else np.nan

    # figure
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    # make space for right-side legend
    fig.subplots_adjust(right=0.82)

    # total curve
    ax.plot(u, H, linewidth=2.2, label="$H(u) = A + B/u + C \\cdot u$")

    # optional component curves (dashed)
    if show_terms:
        ax.plot(u, A_arr, linestyle="--", linewidth=1.4, label="$A$ (eddy diffusion)")
        ax.plot(
            u,
            B_arr,
            linestyle="--",
            linewidth=1.4,
            label="$B/u$ (longitudinal diffusion)",
        )
        ax.plot(
            u,
            C_arr,
            linestyle="--",
            linewidth=1.4,
            label="$C \\cdot u$ (mass transfer)",
        )

    # dominance shading: find where each component is the largest contributor
    comps = np.vstack([A_arr, B_arr, C_arr])  # shape: (3, n)
    dom = np.argmax(comps, axis=0)  # 0:A, 1:B/u, 2:C·u

    # segment the domain where dominance label stays constant
    idx = np.flatnonzero(np.r_[True, np.diff(dom) != 0, True])
    colors = ["C1", "C2", "C3"]  # match matplotlib default cycle for A, B/u, C·u
    labels = ["$A$-dominated", "$B/u$-dominated", "$C \\cdot u$-dominated"]
    seen = set()
    for i0, i1 in zip(idx[:-1], idx[1:]):
        d = dom[i0]
        # light shading matching the corresponding line color
        label = labels[d] if d not in seen else None
        ax.axvspan(u[i0], u[i1 - 1], color=colors[d], alpha=0.08, label=label)
        seen.add(d)

    # mark u_opt and H_min if inside range
    if np.isfinite(u_opt) and (u_min <= u_opt <= u_max) and np.isfinite(H_min):
        ax.axvline(u_opt, linestyle=":", linewidth=1.6)
        ax.plot([u_opt], [H_min], marker="o", markersize=5)
        ax.annotate(
            r"$u_{\mathrm{opt}}$",
            xy=(u_opt, H_min),
            xytext=(6, 8),
            textcoords="offset points",
        )

    # axes & legend
    ax.set_xlabel("Linear velocity, $u$ (mm s$^{-1}$)")
    ax.set_ylabel("Plate height, $H$ (mm)")
    ax.set_title("van Deemter curve with dominance regions")
    ax.grid(True, alpha=0.3)

    # compact legend
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.show()


def van_deemter_particles_plot(
    A0=5.0, B0=1.0, C0=5.0, u_min=0.05, u_max=3.0, show_opt=True
):
    """
    Plot lumped H(u) curves for different particle sizes using typical scalings:
      A(dp) = A0 * (dp/dp_ref)^1
      B(dp) = B0
      C(dp) = C0 * (dp/dp_ref)^2
    Units are pedagogical; axes labeled in mm and mm s^-1 for concreteness.
    """
    # for pedagogical purposes, use 5 um as reference particle size
    dp_ref = 5.0

    # guard rails
    u_min = max(1e-3, float(u_min))
    u_max = max(u_min + 1e-3, float(u_max))
    A0 = float(A0)
    B0 = float(B0)
    C0 = float(C0)
    dp_ref = float(dp_ref)

    # domain
    n = 900
    u = np.linspace(u_min, u_max, n)

    # particle sizes and labels
    particle_sizes = [
        (10.0, "10 $\\mu$m (LC)"),
        (5.0, "5 $\\mu$m (HPLC)"),
        (1.7, "1.7 $\\mu$m (UPLC)"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    # make space for right-side legend
    fig.subplots_adjust(right=0.82)

    for dp, label in particle_sizes:
        scale = dp / dp_ref
        A = A0 * scale
        B = B0
        C = C0 * (scale**2)

        H = A + (B / u) + (C * u)

        # plot and capture color
        (line,) = ax.plot(u, H, linewidth=2.0, label=f"{label}")
        color = line.get_color()

        # optimum markers
        if show_opt and (B > 0) and (C > 0):
            u_opt = np.sqrt(B / C)
            H_min = A + 2.0 * np.sqrt(B * C)
            if np.isfinite(u_opt) and (u_min <= u_opt <= u_max) and np.isfinite(H_min):
                ax.axvline(u_opt, linestyle=":", linewidth=1.2, color=color)
                ax.plot([u_opt], [H_min], marker="o", markersize=4.5, color=color)
                ax.annotate(
                    r"$u_{\mathrm{opt}}$",
                    xy=(u_opt, H_min),
                    xytext=(6, 8),
                    textcoords="offset points",
                    color=color,
                )

    ax.set_xlabel(r"Linear velocity, $u$ (mm s$^{-1}$)")
    ax.set_ylabel(r"Plate height, $H$ (mm)")
    ax.set_title("van Deemter H(u) for different particle sizes")
    ax.grid(True, alpha=0.3)

    # move legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.show()


def _gaussian_sum(t, tR, sigma_t, heights):
    """Sum of Gaussians with identical sigma_t."""
    y = np.zeros_like(t)
    for mu, h in zip(tR, heights):
        y += h * np.exp(-0.5 * ((t - mu) / sigma_t) ** 2)
    return y


def _sigma_time(H_mm, L_mm, u_mm_s, eps=1e-9):
    """
    Convert plate height H (mm) to time-domain sigma (s) via σ_z^2 = H*L and σ_t = σ_z/u.
    """
    H_mm = max(H_mm, eps)
    u_mm_s = max(u_mm_s, eps)
    sigma_z_mm = np.sqrt(H_mm * L_mm)
    return sigma_z_mm / u_mm_s


def plot_chromatograms(
    H_normal_mm=0.050,
    L_mm=150.0,
    u_mm_s=2.0,
    spacing_s=20.0,
    bad_factor=100.0,
    ideal_scale=0.01,
    heights_peaks=[0.6, 1.0, 0.8],
):
    """
    Show three chromatograms (ideal ~ H≈0, normal H, poor high H) with three peaks each.
    Controls:
      - H_normal_mm: 'typical' plate height (mm)
      - L_mm: column length (mm)
      - u_mm_s: linear velocity (mm/s)
      - spacing_s: retention time spacing between adjacent peaks (s)
      - bad_factor: multiplier for 'poor' plate height (H_poor = bad_factor * H_normal)
      - ideal_scale: factor for 'ideal' plate height (H_ideal = ideal_scale * H_normal)
      - heights_peaks: list of relative peak heights (3 values)
    """
    # --- retention times (s)
    t0 = 60.0
    tR = np.array([t0, t0 + spacing_s, t0 + 1.5 * spacing_s], dtype=float)

    # --- map H -> sigma_t (s)
    H_normal = max(H_normal_mm, 1e-6)
    H_ideal = max(ideal_scale * H_normal, 1e-9)  # very small but >0
    H_poor = max(bad_factor * H_normal, 1e-6)

    sigma_t_ideal = _sigma_time(H_ideal, L_mm, u_mm_s)
    sigma_t_normal = _sigma_time(H_normal, L_mm, u_mm_s)
    sigma_t_poor = _sigma_time(H_poor, L_mm, u_mm_s)

    # --- time axis (s), dense enough for very narrow peaks
    t_min = max(0.0, tR[0] - 6 * sigma_t_poor)
    t_max = tR[-1] + 6 * sigma_t_poor
    n_pts = 3000
    t = np.linspace(t_min, t_max, n_pts)

    # --- chromatograms (sum of three Gaussians)
    y_ideal = _gaussian_sum(t, tR, sigma_t_ideal, heights_peaks)
    y_normal = _gaussian_sum(t, tR, sigma_t_normal, heights_peaks)
    y_poor = _gaussian_sum(t, tR, sigma_t_poor, heights_peaks)

    # --- convert x-axis to minutes for display
    t_min_ = t / 60.0
    tR_min = tR / 60.0

    # --- setup figure with shared axes
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.8), sharex=True)
    fig.subplots_adjust(hspace=0.50, left=0.12, right=0.96, top=0.90, bottom=0.12)

    # panels and labels
    panels = [
        (r"Ideal ($H\approx 0$)", y_ideal, H_ideal, sigma_t_ideal),
        (r"Good (normal $H$)", y_normal, H_normal, sigma_t_normal),
        (r"Poor (high $H$)", y_poor, H_poor, sigma_t_poor),
    ]

    for ax, (title, y, H, sig_t) in zip(axes, panels):
        ax.plot(t_min_, y, linewidth=1.8)
        # retention time markers
        for trm in tR_min:
            ax.axvline(trm, linestyle=":", linewidth=1.0, alpha=0.5)

        y_ann = y.max() * 1.25
        y_ann2 = y.max() * 1.1

        # annotate H and N
        N = (L_mm / H) if H > 0 else np.inf
        ax.text(
            0.01,
            y_ann,
            rf"$H={signif(H, 1)}\ \mathrm{{mm}},\ N\approx {signif(N, 2):,.0f}$",
            fontsize=9,
            va="bottom",
        )

        # ------- Baseline resolution (Gaussian): Rs = Δt / [0.5 (w_b1 + w_b2)] with w_b = 4σ_t
        # Identical σ_t within each panel → Rs = Δt / (4 σ_t)
        Rs12 = (tR[1] - tR[0]) / (4.0 * sig_t)
        Rs23 = (tR[2] - tR[1]) / (4.0 * sig_t)

        # arrow positions (in minutes) and annotation heights
        mid12 = 0.5 * (tR_min[0] + tR_min[1])
        mid23 = 0.5 * (tR_min[1] + tR_min[2])

        # draw double-headed arrows between peaks (baseline)
        arrowprops = dict(arrowstyle="<->", lw=1.0, color="0.3", alpha=0.7)
        ax.annotate(
            "",
            xy=(tR_min[0], y_ann2),
            xytext=(tR_min[1], y_ann2),
            arrowprops=arrowprops,
        )
        ax.annotate(
            "",
            xy=(tR_min[1], y_ann),
            xytext=(tR_min[2], y_ann),
            arrowprops=arrowprops,
        )

        # annotate Rs values near midpoints
        ax.text(
            mid12,
            y_ann2,
            rf"$R_s={signif(Rs12, 3)}$",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            mid23,
            y_ann,
            rf"$R_s={signif(Rs23, 3)}$",
            ha="center",
            va="bottom",
            fontsize=9,
        )

        ax.set_ylabel(r"Signal ($-$)")
        ax.set_title(title, fontsize=11)

        # y-lims & grid
        ax.set_ylim(bottom=0, top=y_ann * 1.3)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(r"Time (min)")

    fig.suptitle(
        r"Effect of Plate Height $H$ on Chromatogram (3 peaks, same retention times) $-$ with resolutions $R_s$",
        fontsize=12.5,
    )
    plt.show()


def pencil(ax, x, y=1.0, width=0.8, height=4.2, band_specs=None, tip_h=0.6):
    """
    Draw a pencil-like vertical bar with optional colored bands.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    x, y : float
        Bottom-left reference point for the pencil tip area.
    width, height : float
        Width and height of the main rectangular body (excludes tip).
    band_specs : list[tuple]
        Each tuple: (y0, h, color) relative to the bottom of the body
        (i.e., y + tip_h + y0).
    tip_h : float
        Height of the triangular tip.
    """
    # main body
    body = Rectangle(
        (x, y + tip_h),
        width,
        height,
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(body)

    # tip (triangle)
    tip = Polygon(
        [[x, y + tip_h], [x + width, y + tip_h], [x + width / 2, y]],
        closed=True,
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(tip)

    # colored bands
    if band_specs:
        for y0, h, c in band_specs:
            band = Rectangle(
                (x, y + tip_h + y0),
                width,
                h,
                facecolor=c,
                edgecolor="black",
                linewidth=1.5,
            )
            ax.add_patch(band)


def draw_chromatography_pencils():
    """
    Recreate the pencil diagram from the chromatography fundamentals module,
    with pedagogical labels.
    """
    # First two colors from the default color cycle (typically blue, orange)
    c1, c2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0.5, 9.0)
    ax.set_ylim(0, 6.4)  # a little headroom for the labels
    ax.axis("off")

    # Geometry helpers
    xw = 0.8  # pencil width used for centering text

    # Pencil 1 (bad)
    x1 = 1.0
    pencil(ax, x=x1, band_specs=[(2.6, 1.2, c1), (1.6, 1.0, c2)])
    ax.text(
        x1 + xw / 2,
        6.2,
        "Poor selectivity\nPoor efficiency",
        ha="center",
        va="top",
        fontsize=10,
    )

    # Pencil 2
    x2 = 3.1
    pencil(ax, x=x2, band_specs=[(2.8, 0.8, c1), (1.8, 0.6, c2)])
    ax.text(
        x2 + xw / 2,
        6.2,
        "Poor selectivity\nMedium efficiency",
        ha="center",
        va="top",
        fontsize=10,
    )

    # Pencil 3
    x3 = 5.2
    pencil(ax, x=x3, band_specs=[(2.8, 0.8, c1), (0.8, 0.6, c2)])
    ax.text(
        x3 + xw / 2,
        6.2,
        "Good selectivity\nMedium efficiency",
        ha="center",
        va="top",
        fontsize=10,
    )

    # Pencil 4 (good)
    x4 = 7.3
    pencil(ax, x=x4, band_specs=[(3.0, 0.4, c1), (1.0, 0.2, c2)])
    ax.text(
        x4 + xw / 2,
        6.2,
        "Good selectivity\nGood efficiency",
        ha="center",
        va="top",
        fontsize=10,
    )

    # Axis: Bad -> Good
    ax.plot([0.8, 8.6], [0.5, 0.5], color="black", linewidth=1.5)
    arrow = FancyArrow(
        8.4,
        0.5,
        0.35,
        0,
        width=0.01,
        head_width=0.15,
        head_length=0.3,
        length_includes_head=True,
        color="black",
    )
    ax.add_patch(arrow)
    ax.text(0.8, 0.2, "Bad", fontsize=12, ha="left", va="center")
    ax.text(8.6, 0.2, "Good", fontsize=12, ha="right", va="center")

    plt.tight_layout()
    plt.show()


L_mm = 150.0
beta = 2.0
_TWO_PI = 2.0 * np.pi


def _gauss_area_normalized(x, mu, sigma, area=1.0):
    """Gaussian with constant AREA (not height)."""
    sigma = max(sigma, 1e-9)
    amp = area / (np.sqrt(_TWO_PI) * sigma)
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fig_to_rgb_array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = buf.reshape(h, w, 4)
    return rgba[..., :3].copy()


def make_ldiffusion_gif(
    u_mm_s=2.0,
    D_mm2_s=1.0e-3,
    n_frames=80,
    fps=20,
    path="bterm_travel.gif",
    area=1.0,  # constant peak area (arb. units)
    start_frac=0.02,  # start animation at t = start_frac * (L/u) to avoid singularity,
    title=r"Longitudinal diffusion ($B/u$): band travel & broadening",
):
    """
    Longitudinal diffusion (B/u) GIF:
      - Axial band travels from z=0 to z=L and broadens.
      - Peak area is conserved (height adapts as sigma grows/shrinks).
      - Legend is placed OUTSIDE the axes to the right.
      - Bottom/right margins increased so labels/legend are not cut.
    """
    u = max(u_mm_s, 1e-9)
    D = max(D_mm2_s, 1e-12)
    t_end = L_mm / u

    # Time grid (avoid t=0 to prevent infinite height with area-normalized Gaussian)
    t0 = max(start_frac * t_end, 1e-3)
    t = np.linspace(t0, t_end, n_frames)

    # For consistent y-limits across frames: compute smallest sigma_z -> largest peak height
    sigma_z_min = np.sqrt(beta * D * t0)
    amp_max = area / (np.sqrt(_TWO_PI) * sigma_z_min)  # peak height at t0
    y_top = amp_max * 1.15

    # Axial grid
    z = np.linspace(0, L_mm, 3000)

    frames = []
    for ti in t:
        zc = u * ti
        sigma_z = np.sqrt(beta * D * ti)
        y = _gauss_area_normalized(z, zc, sigma_z, area=area)

        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        # Make room for legend on the right and xlabel at the bottom
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)

        ax.plot(z, y, lw=2.0, label="Peak (B-term only, constant area)")
        ax.axvline(L_mm, ls=":", lw=1.2, color="0.3")
        ax.set_xlim(0, L_mm * 1.02)
        ax.set_ylim(0, y_top)
        ax.set_xlabel(r"Axial position, $z$ (mm)")
        ax.set_ylabel("Normalized band profile (\u2013)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Info box
        info = (
            rf"$u={u:.2f}$ mm s$^{{-1}}$,  $D_m={D:.4f}$ mm$^2$ s$^{{-1}}$"
            + "\n"
            + rf"$t={ti:.2f}$ s,  $\sigma_z(t)=\sqrt{{\beta D_m t}}={sigma_z:.2f}$ mm"
        )
        ax.text(
            0.02,
            0.98,
            info,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"),
        )

        frames.append(Image.fromarray(_fig_to_rgb_array(fig)))
        plt.close(fig)

    if frames:
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=int(2000 / u_mm_s / fps),
            loop=0,
            optimize=True,
        )


def _gauss_pdf(x, mu, sigma):
    sigma = max(sigma, 1e-12)
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def make_brownian_spread_gif(
    path="brownian_spread.gif",
    n_particles=1000,
    D_mm2_s=1.0e-3,  # diffusion coefficient (mm^2/s)
    t_end_s=120.0,  # total time (s)
    dt_s=2,  # time step (s)
    sigma0_mm=0.2,  # initial std dev (mm) around x=0
    x_window_mm=6.0,  # half-width of x-axis window around 0 (for display)
    bins=40,
    fps=24,
    seed=42,
    title="Brownian spreading at a fixed location → Gaussian broadening (Einstein MSD)",
):
    """
    Simulate 1D diffusion (no drift). Initial positions ~ N(0, sigma0^2).
    Each step: x <- x + sqrt(2 D dt) * N(0,1).
    Top panel: particles drawn with vertical jitter for readability.
    Bottom panel: histogram (density=True) + theoretical Gaussian PDF with σ^2 = σ0^2 + 2 D t.
    """
    rng = np.random.default_rng(seed)
    n_steps = max(2, int(np.ceil(t_end_s / dt_s)))
    times = np.linspace(0.0, t_end_s, n_steps)

    # Initial positions
    x = rng.normal(loc=0.0, scale=sigma0_mm, size=n_particles)

    # Precompute static axis limits
    x_min = -x_window_mm
    x_max = x_window_mm
    y_top_hist = 1.0 / (
        np.sqrt(2 * np.pi) * max(sigma0_mm, 1e-12)
    )  # initial peak height; will decrease with t
    y_top_hist *= 1.25

    frames = []
    for t in times:
        # Diffusion step (skip step at t=0)
        if t > 0.0:
            x += np.sqrt(2.0 * D_mm2_s * dt_s) * rng.standard_normal(n_particles)

        # Theoretical distribution at time t
        mu_t = 0.0
        sigma_t = np.sqrt(sigma0_mm**2 + 2.0 * D_mm2_s * t)

        # --- Figure ---
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True)
        fig.subplots_adjust(left=0.12, right=0.92, bottom=0.14, top=0.90, hspace=0.18)

        # Top: molecules as dots with vertical jitter just for separation
        y_jitter = rng.uniform(-0.45, 0.45, size=n_particles)
        ax_top.scatter(x, y_jitter, s=18, alpha=0.9, edgecolor="none")
        ax_top.axvline(0.0, ls=":", lw=1.2, color="0.3")
        ax_top.set_xlim(x_min, x_max)
        ax_top.set_ylim(-0.6, 0.6)
        ax_top.set_yticks([])
        ax_top.set_title(title)
        ax_top.set_ylabel("molecules (vertical jitter)")

        ax_top.text(
            0.02,
            0.90,
            rf"$D_m={D_mm2_s:.4f}$ mm$^2$ s$^{{-1}}$"
            + "\n"
            + rf"$t={t:5.2f}$ s"
            + "\n"
            + rf"$\sigma(t)=\sqrt{{\sigma_0^2+2Dt}}={sigma_t:.2f}$ mm",
            transform=ax_top.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"),
        )

        # Bottom: histogram (density) + theoretical PDF (area=1)
        ax_bot.hist(
            x,
            bins=bins,
            range=(x_min, x_max),
            density=True,
            alpha=0.25,
            color="tab:blue",
            edgecolor="none",
            label="Empirical density",
        )
        xx = np.linspace(x_min, x_max, 1000)
        ax_bot.plot(
            xx, _gauss_pdf(xx, mu_t, sigma_t), lw=2.0, label="Theoretical Gaussian PDF"
        )
        ax_bot.set_ylim(0.0, 2.0)
        ax_bot.set_xlabel(r"Axial position, $x$ (mm)")
        ax_bot.set_ylabel(r"Density [mm$^{-1}$]")
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(loc="upper right", frameon=False)

        # Capture frame
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgba = buf.reshape(h, w, 4)
        frames.append(Image.fromarray(rgba[..., :3].copy()))
        plt.close(fig)

    # Save GIF
    if frames:
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=int(2000 / fps),
            loop=0,
            optimize=True,
        )
    return path
