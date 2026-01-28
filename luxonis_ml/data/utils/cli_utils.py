import ast

import typer


def parse_split_ratio(
    value: str | None,
) -> dict[str, float | int] | None:
    """Parse split ratio argument.

    Expects a Python list (e.g., C{"[0.8, 0.1, 0.1]"}). If values sum to
    1.0, treated as ratios. Otherwise, treated as counts.
    """
    if value is None:
        return None

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as e:
        raise typer.BadParameter(f"Invalid list syntax: {e}") from e

    if not isinstance(parsed, list) or len(parsed) != 3:
        raise typer.BadParameter(
            "Split ratio must be a list of 3 values (train, val, test)."
        )

    if not all(isinstance(v, int | float) for v in parsed):
        raise typer.BadParameter("Split ratio values must be numbers.")

    all_ints = all(isinstance(v, int) for v in parsed)
    all_floats = all(isinstance(v, float) for v in parsed)

    if not (all_ints or all_floats):
        raise typer.BadParameter(
            "Split ratio values must be all integers (counts) "
            "or all floats (ratios), not a mix."
        )

    if all_floats and abs(sum(parsed) - 1.0) >= 1e-6:
        raise typer.BadParameter(
            f"Float ratios must sum to 1.0, but got {sum(parsed):.2f}."
        )

    keys = ["train", "val", "test"]
    return dict(
        zip(
            keys,
            parsed if all_floats else [int(v) for v in parsed],
            strict=True,
        )
    )
