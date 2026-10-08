"""Output-magnitude plots shared by Function 1's weekly notebooks."""

import matplotlib.pyplot as plt


__all__ = ["plot_output_magnitudes"]


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
