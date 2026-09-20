from __future__ import annotations

import json
from typing import Any


def build_notebook(title: str, findings: list[dict[str, Any]], analysis_summary: dict[str, Any]) -> dict[str, Any]:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "\n",
                    "This notebook was generated from the Kaggle Dataset Analysis workspace.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import json\n",
                    "\n",
                    "analysis_summary = json.loads('''" + json.dumps(analysis_summary) + "''')\n",
                    "print('Dataset summary ready for reproduction.')\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    for index, finding in enumerate(findings, start=1):
        notebook["cells"].append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"## Finding {index}: {finding.get('title', 'Analysis result')}\n",
                    "\n",
                    f"{finding.get('summary', '')}\n",
                ],
            }
        )
        notebook["cells"].append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"# Reproducible code for finding {index}\n",
                    f"print({json.dumps(finding.get('summary', ''))})\n",
                ],
            }
        )

    return notebook
