"""Compute complete knowledge-gradient studies without rendering the notebooks."""

import argparse
from pathlib import Path
import sys

stage = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(stage / "src"))
from initial_exploration import load_initial_data
from risk_refinement import ensure_refined_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--functions", nargs="+", type=int, choices=(1, 3, 4, 5, 6, 7, 8),
                        default=[1, 3, 4, 5, 6, 7, 8])
    args = parser.parse_args()
    for function in args.functions:
        data = load_initial_data(stage, function)
        output = stage.parent / f".runtime/function-{function}-risk"
        result = ensure_refined_results(data, function, output,
                                        progress=lambda message: print(message, flush=True))
        point = result["pool"][result["recommended_index"]]
        print(f"Function {function}: proposed input " + "-".join(f"{x:.6f}" for x in point), flush=True)


if __name__ == "__main__":
    main()
