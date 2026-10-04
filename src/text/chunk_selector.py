"""Selection of consecutive 10-K chunks for the FinTok text pipeline."""

import pandas as pd


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"chunk_id", "ticker", "fiscal_year", "item", "posicion", "texto"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing corpus columns: {sorted(missing)}")


def get_context(df: pd.DataFrame, chunk_id: str, window_size: int = 3) -> pd.DataFrame:
    """
    Return a context window around one selected chunk.

    The returned chunks always belong to the same ticker, fiscal year and
    10-K item. For the standard FinTok pipeline, window_size=3.
    At section boundaries the window is shifted so that three chunks are
    returned whenever the section contains at least three chunks.
    """
    _validate_columns(df)

    matches = df.loc[df["chunk_id"] == chunk_id]
    if matches.empty:
        raise ValueError(f"Unknown chunk_id: {chunk_id}")

    target = matches.iloc[0]

    section = (
        df[
            (df["ticker"] == target["ticker"])
            & (df["fiscal_year"] == target["fiscal_year"])
            & (df["item"] == target["item"])
        ]
        .sort_values("posicion")
        .reset_index(drop=True)
    )

    center_idx = int(section.index[section["chunk_id"] == chunk_id][0])

    start = min(
        max(center_idx - window_size // 2, 0),
        max(len(section) - window_size, 0),
    )

    return section.iloc[start:start + window_size].copy()


def build_context_groups(df: pd.DataFrame, window_size: int = 3):
    """
    Split every ticker/year/item section into consecutive non-overlapping
    context groups.

    This is intended for later full-corpus execution. It avoids processing
    nearly identical sliding windows for every overlapping chunk.
    """
    _validate_columns(df)
    groups = []

    ordered = df.sort_values(["ticker", "fiscal_year", "item", "posicion"])

    for (ticker, year, item), section in ordered.groupby(
        ["ticker", "fiscal_year", "item"], sort=False
    ):
        section = section.sort_values("posicion").reset_index(drop=True)

        for start in range(0, len(section), window_size):
            context_df = section.iloc[start:start + window_size].copy()
            groups.append(context_df)

    return groups


def context_to_text(context_df: pd.DataFrame) -> str:
    """Serialize a context DataFrame for insertion into an LLM prompt."""
    return "\n\n".join(
        f"--- CHUNK {row['chunk_id']} ---\n{row['texto']}"
        for _, row in context_df.iterrows()
    )
