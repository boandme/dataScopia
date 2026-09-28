"""Built-in reference datasets that ship with the app (no network, no credentials).

They serve as join partners in the dataset graph: any table with a country column can be enriched
with Gapminder's life expectancy, GDP per capita and population without downloading anything.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

LIBRARY: dict[str, dict[str, Any]] = {
    "library/gapminder": {
        "title": "Gapminder: life expectancy, GDP per capita and population by country (1952-2007)",
        "license": "CC BY 3.0 (Gapminder Foundation)",
        "loader": "gapminder",
    },
}


def _load(name: str) -> pd.DataFrame:
    import plotly.express as px

    frame = getattr(px.data, name)()
    return frame.rename(columns={"pop": "population", "lifeExp": "life_expectancy", "gdpPercap": "gdp_per_capita"}).drop(columns=["iso_num"], errors="ignore")


def list_library() -> list[dict[str, Any]]:
    return [{"id": k, "title": v["title"], "license": v["license"]} for k, v in LIBRARY.items()]


def load_library_frame(library_id: str) -> pd.DataFrame:
    if library_id not in LIBRARY:
        raise KeyError(library_id)
    return _load(LIBRARY[library_id]["loader"])
