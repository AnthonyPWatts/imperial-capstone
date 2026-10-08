"""Summarise submitted outputs against earlier observations without modifying data."""

import numpy as np
import pandas as pd


__all__ = ["summarise_output_observations"]


def summarise_output_observations(
    data: pd.DataFrame, *, input_columns, output_column="y", round_column="source_round"
):
    """Identify the latest recorded submission by round, rather than row order.

    Round zero contains initial observations. Highlight every row in the latest
    positive round and compare it only with earlier rounds. Ranks include all
    observations, with 1 denoting the highest output and ties sharing a rank.
    The report describes observed records and magnitudes, not statistical outliers.

    Return a copy of the data, a positional latest_mask, comparison counts,
    latest_submissions (coordinates, output, rank and notes), and a readable
    report. With only round-zero data, there is no latest submission to highlight.
    """
    input_columns = list(input_columns)
    required = [*input_columns, output_column, round_column]
    if not input_columns or len(set(required)) != len(required):
        raise ValueError("Supply distinct input, output and round column names.")
    if not set(required).issubset(data.columns):
        raise ValueError("Data must contain the requested input, output and round columns.")
    observations = data.copy(deep=True)
    numeric = observations[required].to_numpy(dtype=float)
    if len(observations) == 0 or not np.isfinite(numeric).all():
        raise ValueError("Observations must contain finite inputs, outputs and rounds.")
    rounds = observations[round_column].to_numpy(dtype=float)
    if np.any(rounds < 0) or np.any(rounds != np.floor(rounds)):
        raise ValueError("Source rounds must be non-negative integers.")
    values = observations[output_column].to_numpy(dtype=float)
    latest_round = int(rounds.max())
    latest_mask = rounds == latest_round if latest_round > 0 else np.zeros(len(values), dtype=bool)
    prior_values = values[~latest_mask]
    ranks = pd.Series(values).rank(ascending=False, method="min").to_numpy(dtype=int)
    columns = (["observation_id"] if "observation_id" in observations else [])
    columns += [round_column, *input_columns, output_column]
    latest_submissions = observations.loc[latest_mask, columns].copy()
    latest_submissions["output_rank"] = ranks[latest_mask]
    notes = []

    if latest_round == 0:
        lines = [
            "Initial observations only: no submitted result to highlight.",
            f"{len(values)} outputs; observed range: [{values.min():.12g}, {values.max():.12g}].",
        ]
    else:
        count = int(latest_mask.sum())
        lines = [
            f"Latest recorded submission: round {latest_round} ({count} new {'output' if count == 1 else 'outputs'}).",
            f"Compared with {len(prior_values)} earlier observations; rank 1 = highest of all {len(values)} outputs.",
        ]
        if len(prior_values):
            lines.append(f"Earlier output range: [{prior_values.min():.12g}, {prior_values.max():.12g}].")

        for position in np.flatnonzero(latest_mask):
            row = observations.iloc[position]
            value = values[position]
            label = str(row["observation_id"]) if "observation_id" in observations else f"Row {position + 1}"
            sign = "negative" if value < 0 else "positive" if value > 0 else "exact zero"
            comparison_notes = []
            if len(prior_values) == 0:
                comparison_notes.append("No earlier outputs for comparison.")
            elif value < prior_values.min():
                comparison_notes.append(
                    "New lowest observed output." if value == values.min()
                    else "Below the earlier output minimum."
                )
            elif value > prior_values.max():
                comparison_notes.append(
                    "New highest observed output." if value == values.max()
                    else "Above the earlier output maximum."
                )
            elif value == prior_values.max():
                comparison_notes.append("Matches the earlier highest output.")
            elif value == prior_values.min():
                comparison_notes.append("Matches the earlier lowest output.")
            else:
                comparison_notes.append("Within the earlier output range.")

            magnitude = abs(value)
            prior_magnitudes = np.abs(prior_values)
            nonzero_prior = prior_magnitudes[prior_magnitudes > 0]
            if magnitude == 0:
                comparison_notes.append("Exactly zero, rather than a rounded small value.")
            elif len(prior_values) and len(nonzero_prior) == 0:
                comparison_notes.append("First non-zero output.")
            elif len(nonzero_prior):
                log_magnitude = np.log10(magnitude)
                largest_log = np.log10(nonzero_prior.max())
                smallest_log = np.log10(nonzero_prior.min())
                if log_magnitude > largest_log:
                    difference = log_magnitude - largest_log
                    description = (
                        "Largest absolute magnitude so far" if magnitude == np.abs(values).max()
                        else "Above the earlier largest absolute magnitude"
                    )
                    comparison_notes.append(
                        f"{description} ({10.0 ** difference:.3g} times the earlier largest)."
                        if difference < 6 else
                        f"{description} ({difference:.1f} orders above the earlier largest)."
                    )
                elif log_magnitude < smallest_log:
                    difference = smallest_log - log_magnitude
                    description = (
                        "Smallest non-zero magnitude so far" if magnitude == np.abs(values[values != 0]).min()
                        else "Below the earlier smallest non-zero magnitude"
                    )
                    comparison_notes.append(
                        f"{description} ({10.0 ** difference:.3g} times smaller than the earlier smallest)."
                        if difference < 6 else
                        f"{description} ({difference:.1f} orders below the earlier smallest)."
                    )
                elif largest_log - log_magnitude >= 6:
                    comparison_notes.append(
                        f"Magnitude is {largest_log - log_magnitude:.1f} orders below the earlier largest."
                    )

            note = " ".join(comparison_notes)
            notes.append(note)
            coordinates = ", ".join(f"{name} = {float(row[name]):.12g}" for name in input_columns)
            lines.extend([
                "",
                f"{label}: {output_column} = {value:.12g} ({sign}); rank {ranks[position]}/{len(values)}.",
                f"Inputs: {coordinates}",
                note,
            ])

    latest_submissions["comparison"] = notes
    return {
        "data": observations,
        "input_columns": input_columns,
        "output_column": output_column,
        "round_column": round_column,
        "latest_round": latest_round if latest_round > 0 else None,
        "latest_mask": latest_mask,
        "prior_count": len(prior_values),
        "latest_submissions": latest_submissions,
        "report": "\n".join(lines),
    }
