import numpy as np
import matplotlib.pyplot as plt


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
