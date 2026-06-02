"""Shared types for the batch-analysis tab."""

from __future__ import annotations

from typing import TypedDict

from PyQt5.QtWidgets import QTreeWidgetItem


class FileInfo(TypedDict):
    name: str
    path: str
    folder: str
    merge: bool
    group: int
    progress: int
    tree_item: QTreeWidgetItem
