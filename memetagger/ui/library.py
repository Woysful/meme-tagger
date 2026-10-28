"""Tag library: every known tag with its kind and usage; rename, merge, change kind, delete (also in files)."""
from __future__ import annotations
from collections import Counter
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)
from ..tags import KIND_LABELS, KINDS, clean_tag
from .common import hint, kind_color
from .widgets import TagLineEdit


def ask_tag(parent, title: str, label: str, initial: str, words_fn) -> str | None:
    d = QDialog(parent); d.setWindowTitle(title)
    lay = QVBoxLayout(d)
    lay.addWidget(QLabel(label))
    edit = TagLineEdit(words_fn); edit.setText(initial); edit.selectAll(); edit.setMinimumWidth(360)
    lay.addWidget(edit)
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    bb.accepted.connect(d.accept); bb.rejected.connect(d.reject)
    lay.addWidget(bb)
    if d.exec() != QDialog.DialogCode.Accepted:
        return None
    return clean_tag(edit.text().strip()) or None


class NumItem(QTableWidgetItem):
    def __init__(self, n: int):
        super().__init__(str(n) if n else "")
        self.n = n
        self.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self, other):
        return self.n < getattr(other, "n", 0)


class TagLibraryDialog(QDialog):
    COLS = ["Tag", "Kind", "Files", "In list", "Franchise"]

    def __init__(self, main):
        super().__init__(main)
        self.main, self.store = main, main.store
        self.setWindowTitle("Tag library"); self.resize(820, 620)
        lay = QVBoxLayout(self); lay.setSpacing(10)
        top = QHBoxLayout()
        self.filter = QLineEdit(); self.filter.setPlaceholderText("Find a tag…"); self.filter.setClearButtonEnabled(True)
        self.kind_filter = QComboBox(); self.kind_filter.addItem("All kinds", None)
        for k in KINDS:
            self.kind_filter.addItem(KIND_LABELS[k], k)
        self.count = hint("", wrap=False)
        top.addWidget(self.filter, 1); top.addWidget(self.kind_filter); top.addWidget(self.count)
        lay.addLayout(top)
        self.table = QTableWidget(0, len(self.COLS)); self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().hide(); self.table.setShowGrid(False); self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(self.COLS)):
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.doubleClicked.connect(lambda *_: self.rename())
        lay.addWidget(self.table, 1)

        act = QHBoxLayout()
        self.rename_btn = QPushButton("Rename / merge…"); self.rename_btn.clicked.connect(self.rename)
        self.kind_box = QComboBox()
        for k in KINDS:
            self.kind_box.addItem(KIND_LABELS[k], k)
        self.kind_btn = QPushButton("Set kind"); self.kind_btn.clicked.connect(self.set_kind)
        self.delete_btn = QPushButton("Delete…"); self.delete_btn.setObjectName("danger"); self.delete_btn.clicked.connect(self.delete)
        act.addWidget(self.rename_btn); act.addSpacing(12); act.addWidget(QLabel("Kind:")); act.addWidget(self.kind_box)
        act.addWidget(self.kind_btn); act.addStretch(1); act.addWidget(self.delete_btn)
        lay.addLayout(act)
        dirs = main.library_dirs()
        self.in_files = QCheckBox("Also change files in " + (", ".join(str(d) for d in dirs[:2]) + ("…" if len(dirs) > 2 else "")
                                                             if dirs else "the library (no library folder set)"))
        self.in_files.setChecked(bool(dirs)); self.in_files.setEnabled(bool(dirs))
        lay.addWidget(self.in_files)
        lay.addWidget(hint("Renaming to an existing tag merges both. File changes are written to the undo journal, "
                           "so “Undo” in the main window reverts them."))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close); bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.filter.textChanged.connect(self.apply_filter); self.kind_filter.currentIndexChanged.connect(self.apply_filter)
        self.table.itemSelectionChanged.connect(self.on_select)
        self.reload()

    # ---- data ----
    def reload(self):
        in_list = Counter(t for r in self.main.model.rows for t in r.item.tags)
        rows = {t: (k, n) for t, k, n in self.store.all_tags()}
        for t in in_list:
            rows.setdefault(t, (self.main.kind_of(t) or "other", 0))
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for i, (t, (k, n)) in enumerate(sorted(rows.items(), key=lambda kv: (-kv[1][1] - in_list[kv[0]], kv[0]))):
            kind = QTableWidgetItem("● " + KIND_LABELS.get(k, "Other")); kind.setForeground(kind_color(k)); kind.setData(Qt.ItemDataRole.UserRole, k)
            fr = self.store.franchise_of(t) if k == "character" else None
            fr = fr or (self.main.db.franchise_of(t) if k == "character" else None)
            name = QTableWidgetItem(t); name.setData(Qt.ItemDataRole.UserRole, t)
            for c, cell in enumerate((name, kind, NumItem(n), NumItem(in_list[t]), QTableWidgetItem(fr or ""))):
                self.table.setItem(i, c, cell)
        self.table.setSortingEnabled(True)
        self.apply_filter(); self.on_select()

    def apply_filter(self, *_):
        q, kind = self.filter.text().strip().lower(), self.kind_filter.currentData()
        shown = 0
        for i in range(self.table.rowCount()):
            t = self.table.item(i, 0).text()
            k = self.table.item(i, 1).data(Qt.ItemDataRole.UserRole)
            hide = bool((q and q not in t) or (kind and k != kind))
            self.table.setRowHidden(i, hide)
            shown += not hide
        self.count.setText(f"{shown} of {self.table.rowCount()} tags")

    def selected(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.table.item(r, 0).text() for r in rows if not self.table.isRowHidden(r)]

    def on_select(self):
        sel = self.selected()
        for b in (self.rename_btn, self.kind_btn, self.delete_btn):
            b.setEnabled(bool(sel))
        self.rename_btn.setText("Merge…" if len(sel) > 1 else "Rename / merge…")
        if len(sel) == 1:
            k = self.store.kind_of(sel[0]) or self.main.kind_of(sel[0]) or "other"
            self.kind_box.setCurrentIndex(max(0, self.kind_box.findData(k)))

    # ---- actions ----
    def rename(self):
        sel = self.selected()
        if not sel:
            return
        label = (f"New name for “{sel[0]}”." if len(sel) == 1 else f"Merge {len(sel)} tags into:") + \
            "\nAn existing tag name merges them."
        new = ask_tag(self, "Rename tag", label, sel[0], self.main.tag_words)
        if not new or (len(sel) == 1 and new == sel[0]):
            return
        mapping = {t: new for t in sel if t != new}
        if self.main.apply_tag_mapping(mapping, self.in_files.isChecked()):
            for old in mapping:
                self.store.rename(old, new)
            self.main.after_vocabulary_change()
            self.reload()

    def delete(self):
        sel = self.selected()
        if not sel:
            return
        names = ", ".join(sel[:6]) + ("…" if len(sel) > 6 else "")
        if QMessageBox.question(self, "Delete tags", f"Delete {names} from the vocabulary"
                                + (" and remove it from the files?" if self.in_files.isChecked() else "?")) \
                != QMessageBox.StandardButton.Yes:
            return
        if self.main.apply_tag_mapping({t: None for t in sel}, self.in_files.isChecked(), confirm=False):
            for t in sel:
                self.store.delete(t)
            self.main.after_vocabulary_change()
            self.reload()

    def set_kind(self):
        kind = self.kind_box.currentData()
        for t in self.selected():
            self.store.set_kind(t, kind)
            for r in self.main.model.rows:
                if t in r.item.tags:
                    r.item.kinds[t] = kind
        self.main.after_vocabulary_change()
        self.reload()
