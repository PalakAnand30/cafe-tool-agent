"""Validate our output-free nbformat 4.5 notebooks without executing GPU cells."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from cafe_agent.config import StrictModel


class NotebookCell(StrictModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)
    cell_type: Literal["code", "markdown"]
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    metadata: dict[str, Any]
    source: list[str]
    outputs: list[dict] | None = None
    execution_count: None = None

    @model_validator(mode="after")
    def check_unexecuted(self) -> "NotebookCell":
        if self.cell_type == "code":
            if self.outputs != []:
                raise ValueError("Notebook code cells must be output-free")
            compile("".join(self.source), f"<cell {self.id}>", "exec")
        elif self.outputs is not None:
            raise ValueError("Markdown cannot contain execution outputs")
        return self


class Notebook(StrictModel):
    nbformat: Literal[4]
    nbformat_minor: Literal[5]
    metadata: dict[str, Any]
    cells: list[NotebookCell] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cells(self) -> "Notebook":
        if len({cell.id for cell in self.cells}) != len(self.cells):
            raise ValueError("Duplicate notebook cell IDs")
        return self


def validate_notebook(path: Path) -> dict:
    notebook = Notebook.model_validate(json.loads(path.read_text()))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "format": "nbformat 4.5",
        "cells": len(notebook.cells),
        "compiled_code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
        "executed_gpu_cells": 0,
        "outputs": 0,
    }
