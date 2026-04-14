from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "benchmarks"
CORRUPTION_DIR = BENCHMARK_ROOT / "corruptions"


def _copy_with_originals(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in ["product_name_raw", "publisher_raw", "grade"]:
        if column in result.columns:
            result[f"original_{column}"] = result[column]
    return result


def _blank_series(length: int) -> pd.Series:
    return pd.Series([""] * length, dtype="object")


def _parse_grade_token(value: object) -> str | None:
    if pd.isna(value):
        return None
    token = str(value).strip()
    if not token:
        return None
    upper = token.upper()
    if upper in {"TK", "PK", "PREK", "K"}:
        return upper
    digits = "".join(ch for ch in token if ch.isdigit())
    return digits or None


def _corrupt_grade(row: pd.Series) -> str:
    current = _parse_grade_token(row.get("grade"))
    if current is None:
        return ""
    if current == "TK":
        return "8"
    if current == "K":
        return "8"
    try:
        value = int(current)
    except ValueError:
        return ""
    return "10" if value <= 5 else "1"


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    description: str

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class CleanReferenceSpec(CorruptionSpec):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        result = _copy_with_originals(df)
        result["corruption_type"] = self.name
        result["corruption_notes"] = self.description
        return result


class SwapTitlePublisherSpec(CorruptionSpec):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        result = _copy_with_originals(df)
        original_title = result["product_name_raw"].copy()
        result["product_name_raw"] = result["publisher_raw"].fillna("")
        result["publisher_raw"] = original_title.fillna("")
        result["corruption_type"] = self.name
        result["corruption_notes"] = self.description
        return result


class TitleInPublisherSpec(CorruptionSpec):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        result = _copy_with_originals(df)
        result["publisher_raw"] = result["product_name_raw"].fillna("")
        result["product_name_raw"] = _blank_series(len(result))
        result["corruption_type"] = self.name
        result["corruption_notes"] = self.description
        return result


class PublisherInTitleSpec(CorruptionSpec):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        result = _copy_with_originals(df)
        result["product_name_raw"] = result["publisher_raw"].fillna("")
        result["publisher_raw"] = _blank_series(len(result))
        result["corruption_type"] = self.name
        result["corruption_notes"] = self.description
        return result


class GradeCorruptedSpec(CorruptionSpec):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        result = _copy_with_originals(df)
        result["grade"] = result.apply(_corrupt_grade, axis=1)
        result["corruption_type"] = self.name
        result["corruption_notes"] = self.description
        return result


CORRUPTION_SPECS: tuple[CorruptionSpec, ...] = (
    CleanReferenceSpec(
        name="clean_reference",
        description="Unmodified reference copy of the source benchmark for comparison.",
    ),
    SwapTitlePublisherSpec(
        name="title_publisher_swapped",
        description="Swap product_name_raw and publisher_raw.",
    ),
    TitleInPublisherSpec(
        name="title_in_publisher_title_blank",
        description="Move product_name_raw into publisher_raw and blank product_name_raw.",
    ),
    PublisherInTitleSpec(
        name="publisher_in_title_publisher_blank",
        description="Move publisher_raw into product_name_raw and blank publisher_raw.",
    ),
    GradeCorruptedSpec(
        name="grade_corrupted_or_missing",
        description="Replace grade with a mismatched or blank value derived from collection_grades.",
    ),
)


def build_corruption_assets(source_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    assets: dict[str, pd.DataFrame] = {}
    for spec in CORRUPTION_SPECS:
        assets[spec.name] = spec.apply(source_df)
    return assets
