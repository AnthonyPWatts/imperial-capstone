"""Output-magnitude plots shared by Function 1's weekly notebooks."""

import matplotlib.pyplot as plt
import numpy as np


__all__ = ["plot_output_magnitudes", "plot_output_number_line"]


def plot_output_magnitudes(y, near_zero_exponent=-15):
    """Compare output values on linear and logarithmic magnitude scales.

    Pass a pandas Series; its index labels identify the observations on the
    logarithmic plot. The near-zero boundary describes the plot and does not
    round or modify outputs. Exact zeros appear only in the linear view.
    Returns a closed figure for a single inline notebook display.
    """
    # This boundary describes the plot; it is not a rule for rounding outputs to zero.
    near_zero_limit = 10.0 ** near_zero_exponent
    near_zero_count = (y.abs() < near_zero_limit).sum()
    near_zero_positive_count = ((y > 0) & (y < near_zero_limit)).sum()
    near_zero_negative_count = ((y < 0) & (y > -near_zero_limit)).sum()
    zero_count = (y == 0).sum()

    fig, (linear_ax, log_ax) = plt.subplots(1, 2, figsize=(14, 5))

    # Use shared bins and stack the counts by sign: a bin crossing zero can contain both.
    # Include exact zeros as a separate group so every observation remains in this view.
    linear_ax.hist(
        [y[y > 0], y[y < 0], y[y == 0]],
        bins=20, stacked=True, edgecolor="black",
        color=["tab:blue", "tab:orange", "grey"],
        label=["Positive", "Negative", "Exact zero"],
    )
    linear_ax.legend(loc="center left")
    linear_ax.axvline(0, color="grey", linestyle=":")
    linear_ax.set_title("Most outputs are crowded near zero")
    linear_ax.set_xlabel("Output value (linear scale)")
    linear_ax.set_ylabel("Number of observations")
    linear_ax.text(
        0.05, 0.95,
        f"{near_zero_count} of {len(y)} observations near zero\n"
        + f"{near_zero_positive_count} positive, {near_zero_negative_count} negative"
        + (f", {zero_count} exact zero" if zero_count else "")
        + "\n" + rf"$|y| < 10^{{{near_zero_exponent}}}$",
        transform=linear_ax.transAxes, va="top",
        bbox=dict(facecolor="white", edgecolor="lightgrey", pad=8),
    )

    # A logarithmic axis cannot include zero. Absolute values give distance from zero;
    # colour and marker shape retain the original sign. Each row is one observation.
    for label, outputs, colour, marker in [
        ("Positive", y[y > 0], "tab:blue", "o"),
        ("Negative", y[y < 0], "tab:orange", "^"),
    ]:
        log_ax.scatter(
            outputs.abs(), outputs.index,
            color=colour, marker=marker, s=65, label=label,
        )

    # Plot the magnitudes themselves; the axis supplies the logarithmic spacing
    # and labels them as powers of ten rather than showing their logarithms.
    log_ax.set_xscale("log")
    log_ax.axvline(
        near_zero_limit, color="grey", linestyle=":",
        label="Chosen near-zero boundary",
    )
    log_ax.set_title("Their distances from zero vary enormously\nrelative to each other")
    log_ax.set_xlabel(r"$\leftarrow$ Closer to zero     |     Absolute output $|y|$ (log scale)")
    log_ax.set_ylabel("Data row index")
    log_ax.set_yticks(y.index)
    log_ax.grid(axis="x", alpha=0.25)
    log_ax.legend()

    print(f"Exact zeros: {zero_count} (included on the left; omitted from the log view).")
    print("The dotted lines mark zero on the left and the near-zero boundary on the right.")
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_output_number_line(review, *, title=None, save_to=None):
    """Summarise measured outputs on mirrored log axes with one shared baseline.

    The centre divides the signs; logarithmic axes approach but never reach zero.
    Actual zeros sit in the centre. Nearby magnitudes may overlap, so the panel
    headings also give observation counts. The input data are left unchanged.
    """
    observations = review["data"]
    values = observations[review["output_column"]].to_numpy(dtype=float)
    latest = np.asarray(review["latest_mask"], dtype=bool)
    magnitudes = np.abs(values)
    nonzero = magnitudes > 0
    labels = observations["observation_id"].astype(str).to_numpy()

    # Matching limits preserve comparisons between signs. Stop one decade below
    # the smallest non-zero magnitude so that even the latest tiny value is kept.
    logs = np.log10(magnitudes[nonzero])
    lower_exponent = int(np.floor(logs.min())) - 1 if len(logs) else -1
    upper_exponent = int(np.ceil(logs.max())) if len(logs) else 0
    lower, upper = 10.0 ** lower_exponent, 10.0 ** upper_exponent
    tick_step = max(1, int(np.ceil((upper_exponent - lower_exponent) / 5 / 10)) * 10)
    exponents = np.arange(np.ceil(lower_exponent / tick_step) * tick_step,
                          upper_exponent, tick_step).astype(int)
    exponents = np.unique(np.r_[exponents, upper_exponent])
    ticks = 10.0 ** exponents

    figure = plt.figure(figsize=(14, 4.5))
    grid = figure.add_gridspec(1, 3, width_ratios=[1, 0.14, 1], wspace=0.04)
    negative_axis = figure.add_subplot(grid[0, 0])
    zero_axis = figure.add_subplot(grid[0, 1], sharey=negative_axis)
    positive_axis = figure.add_subplot(grid[0, 2], sharey=negative_axis)
    zero_axis.set(xlim=(0, 1))
    zero_axis.set_axis_off()
    zero_axis.axvline(0.5, color="grey", linestyle=":", linewidth=1)
    zero_axis.text(0.5, 0.85, "Virtual\nzero", transform=zero_axis.transAxes,
                   ha="center", va="center", fontsize=9,
                   bbox=dict(facecolor="white", edgecolor="none", pad=3))

    positions = np.zeros(len(values))
    for axis, sign_mask, sign_name in (
        (negative_axis, values < 0, "Negative"),
        (positive_axis, values > 0, "Positive"),
    ):
        axis.set_xscale("log")
        axis.set_xlim((upper, lower) if sign_name == "Negative" else (lower, upper))
        prefix = "-" if sign_name == "Negative" else "+"
        axis.set_xticks(ticks, [rf"${prefix}10^{{{exponent}}}$" for exponent in exponents])
        axis.set_xticks([], minor=True)
        axis.set_title(f"{sign_name} outputs ({int(sign_mask.sum())})", fontsize=12, pad=12)
        earlier = sign_mask & ~latest
        if earlier.any():
            axis.scatter(magnitudes[earlier], positions[earlier], s=55, marker="o",
                         color="tab:blue", label="Previous observations",
                         zorder=3, clip_on=False)
        new = sign_mask & latest
        if new.any():
            axis.scatter(magnitudes[new], positions[new], s=190, marker="*", color="tab:red",
                         edgecolors="#263238", linewidths=0.7,
                         label=f"Latest submission (round {review['latest_round']})", zorder=4,
                         clip_on=False)
        axis.grid(False)
        axis.yaxis.set_visible(False)
        axis.set_facecolor("white")
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.spines["bottom"].set_position(("data", 0))
        axis.spines["bottom"].set_color("#64748b")

    zeros = ~nonzero
    if zeros.any():
        zero_axis.scatter(np.full(int(zeros.sum()), 0.5), np.zeros(int(zeros.sum())),
                          color="grey", marker="D", s=50, label="Exact zero")
        latest_zeros = zeros & latest
        if latest_zeros.any():
            zero_axis.scatter(np.full(int(latest_zeros.sum()), 0.5), np.zeros(int(latest_zeros.sum())),
                              color="tab:red", marker="*", s=190,
                              label=f"Latest submission (round {review['latest_round']})")

    negative_axis.set(ylim=(-0.5, 1), yticks=[])
    zero_axis.axhline(0, color="#64748b", linewidth=0.8)
    # Both signs share the same observation styling and a single legend.
    legend_items = {}
    for axis in (negative_axis, positive_axis, zero_axis):
        handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, axis_labels):
            legend_items.setdefault(label, handle)
    figure.legend(list(legend_items.values()), list(legend_items),
                  loc="upper left", bbox_to_anchor=(0.10, 0.73), fontsize=9)
    figure.suptitle(title or "Function 1: observed outputs on a number line", fontsize=15, y=0.98)
    latest_notes = [f"{labels[i]}: y = {values[i]:+.3e}; rank {1 + int((values > values[i]).sum())}/{len(values)} (1 = highest output)"
                    for i in np.flatnonzero(latest)]
    figure.text(0.5, 0.91, "\n".join(latest_notes), ha="center", fontsize=10)
    figure.text(0.5, 0.04,
                "Signed tick labels; both halves use the same logarithmic magnitude scale. Better signed outputs run left to right.\n"
                f"The centre is a sign divider; log axes stop at {lower:.0e}. Exact zeros: {int(zeros.sum())}.",
                ha="center", fontsize=9)
    figure.subplots_adjust(left=0.10, right=0.98, top=0.73, bottom=0.27)
    if save_to is not None:
        figure.savefig(save_to, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return figure
