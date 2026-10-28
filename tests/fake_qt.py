"""Минимальная заглушка PySide6 для проверки ЛОГИКИ контроллера без Qt и дисплея
(проверяет наш код, а не корректность вызовов реального Qt)."""
import sys, threading, types


class _Enum:
    def __init__(self, n): self.n = n
    def __or__(self, o): return self
    def __eq__(self, o): return isinstance(o, _Enum) and o.n == self.n
    def __hash__(self): return hash(self.n)
    def __call__(self, *a, **k): return Stub()
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        return _Enum(n)


class _Meta(type):
    def __getattr__(cls, n):
        if n.startswith("__"): raise AttributeError(n)
        return _Enum(n)


class _Call:
    """И метод-заглушка, и сигнал (connect/emit)."""
    def __init__(self, owner=None): self.slots, self.owner = [], owner
    def __call__(self, *a, **k): return Stub()
    def connect(self, f): self.slots.append(f)
    def disconnect(self, *a): self.slots.clear()
    def emit(self, *a):
        Stub.current_sender = self.owner
        for f in list(self.slots): f(*a)
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        return _Call()


class Stub(metaclass=_Meta):
    current_sender = None
    def __init__(self, *a, **k): pass
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        c = _Call(self); self.__dict__[n] = c; return c
    def sender(self): return Stub.current_sender
    def closeEvent(self, e): pass          # super().closeEvent() реального QMainWindow


class Signal:
    def __init__(self, *t): pass
    def __set_name__(self, owner, name): self.name = name
    def __get__(self, obj, owner=None):
        if obj is None: return self
        return obj.__dict__.setdefault("_sig_" + self.name, _Call(obj))


class QSettings(Stub):
    store = {}
    def value(self, k, d=None, t=None): return QSettings.store.get(k, d)
    def setValue(self, k, v): QSettings.store[k] = v


class QLineEdit(Stub):
    def __init__(self, text=""): self._t = text
    def text(self): return self._t
    def setText(self, t): self._t = t; self.textChanged.emit(t)


class QCheckBox(Stub):
    def __init__(self, text=""): self._c, self._en = False, True
    def isChecked(self): return self._c
    def setChecked(self, c):
        if c != self._c: self._c = c; self.toggled.emit(c)
    def setEnabled(self, e): self._en = e
    def isEnabled(self): return self._en


class QPushButton(Stub):
    def __init__(self, text=""): self._t, self._en, self._name = text, True, ""
    def text(self): return self._t
    def setText(self, t): self._t = t
    def setEnabled(self, e): self._en = e
    def isEnabled(self): return self._en
    def setObjectName(self, n): self._name = n


class QLabel(Stub):
    def __init__(self, text=""): self._t, self._props = text, {}
    def setProperty(self, k, v): self._props[k] = v
    def property(self, k): return self._props.get(k)
    def text(self): return self._t
    def setText(self, t): self._t = t


class _Spin(Stub):
    def __init__(self): self._v = 0
    def value(self): return self._v
    def setValue(self, v): self._v = v


QSpinBox = type("QSpinBox", (_Spin,), {})
QDoubleSpinBox = type("QDoubleSpinBox", (_Spin,), {})


class QProgressBar(Stub):
    def __init__(self): self._r, self._v = (0, 1), 0
    def setRange(self, a, b): self._r = (a, b)
    def setValue(self, v): self._v = v
    def value(self): return self._v


class QComboBox(Stub):
    def __init__(self): self.items, self._text, self._block = [], "", False
    def addItem(self, t, d=None):
        self.items.append((t, d))
        if not self._text: self._text = t
    def addItems(self, l):
        for t in l: self.addItem(t)
    def clear(self): self.items.clear(); self._text = ""
    def count(self): return len(self.items)
    def findData(self, d): return next((i for i, (_, x) in enumerate(self.items) if x == d), -1)
    def setCurrentIndex(self, i):
        if 0 <= i < len(self.items): self._text = self.items[i][0]
    def currentIndex(self): return next((i for i, (t, _) in enumerate(self.items) if t == self._text), -1)
    def currentText(self): return self._text
    def setCurrentText(self, t):
        self._text = t
        if not self._block: self.currentTextChanged.emit(t)
    def currentData(self): return next((d for t, d in self.items if t == self._text), None)
    def blockSignals(self, b): self._block = b
    def itemTexts(self): return [t for t, _ in self.items]


class QTableWidgetItem(Stub):
    def __init__(self, text=""):
        self._t, self._tip, self._bg, self.table, self._r, self._c = text, "", None, None, -1, -1
    def text(self): return self._t
    def setText(self, t):
        self._t = t
        if self.table is not None: self.table.itemChanged.emit(self)
    def setToolTip(self, t): self._tip = t
    def toolTip(self): return self._tip
    def setBackground(self, b): self._bg = b
    def row(self): return self._r
    def column(self): return self._c


class QTableWidget(Stub):
    def __init__(self, r=0, c=0): self.cells, self.hidden = {}, {}; self._rows = r
    def setRowCount(self, n):
        self._rows = n
        for k in [k for k in self.cells if k[0] >= n]: del self.cells[k]
    def rowCount(self): return self._rows
    def setHorizontalHeaderLabels(self, l): self.header = list(l)
    def setItem(self, r, c, it): it.table, it._r, it._c = self, r, c; self.cells[(r, c)] = it
    def item(self, r, c): return self.cells.get((r, c))
    def setRowHidden(self, r, h): self.hidden[r] = h
    def isRowHidden(self, r): return self.hidden.get(r, False)


class QStackedWidget(Stub):
    def __init__(self): self._i = 0
    def setCurrentIndex(self, i): self._i = i
    def currentIndex(self): return self._i


class QTimer(Stub):
    def __init__(self, **k):
        self.started = False
        if "timeout" in k: self.timeout.connect(k["timeout"])
    def start(self, *a): self.started = True
    def stop(self): self.started = False
    @staticmethod
    def singleShot(*a): pass


class QThread(Stub):
    finished = Signal()
    def __init__(self): self._t, self._int = None, False
    def start(self):
        def go():
            try: self.run()
            finally: self.finished.emit()
        self._t = threading.Thread(target=go, daemon=True); self._t.start()
    def isRunning(self): return bool(self._t and self._t.is_alive())
    def wait(self, ms=None):
        if self._t and self._t is not threading.current_thread(): self._t.join()
        return True
    def requestInterruption(self): self._int = True
    def isInterruptionRequested(self): return self._int


class QSplitter(Stub):
    def __init__(self, *a): self._sizes = [0, 0]
    def setSizes(self, l): self._sizes = list(l)
    def sizes(self): return list(self._sizes)


class QTabWidget(Stub):
    def __init__(self): self._i = 0
    def currentIndex(self): return self._i
    def setCurrentIndex(self, i): self._i = i


class QMessageBox(Stub):
    calls = []
    answer = _Enum("Yes")
    @staticmethod
    def _rec(kind, *a): QMessageBox.calls.append((kind, a[1:])); return QMessageBox.answer
    information = staticmethod(lambda *a: QMessageBox._rec("information", *a))
    warning = staticmethod(lambda *a: QMessageBox._rec("warning", *a))
    critical = staticmethod(lambda *a: QMessageBox._rec("critical", *a))
    question = staticmethod(lambda *a: QMessageBox._rec("question", *a))


class QDesktopServices(Stub):
    opened = []
    @staticmethod
    def openUrl(u): QDesktopServices.opened.append(u); return True


class QUrl(Stub):
    @staticmethod
    def fromLocalFile(p): return p


class QFileDialog(Stub):
    getOpenFileNames = staticmethod(lambda *a, **k: ([], ""))
    getExistingDirectory = staticmethod(lambda *a, **k: "")


class QApplication(Stub):
    @staticmethod
    def instance(): return Stub()
    @staticmethod
    def processEvents(): pass


def install():
    explicit = {"QSettings": QSettings, "QLineEdit": QLineEdit, "QCheckBox": QCheckBox, "QPushButton": QPushButton, "QLabel": QLabel,
                "QSpinBox": QSpinBox, "QDoubleSpinBox": QDoubleSpinBox, "QProgressBar": QProgressBar, "QComboBox": QComboBox,
                "QTableWidgetItem": QTableWidgetItem, "QTableWidget": QTableWidget, "QStackedWidget": QStackedWidget,
                "QTimer": QTimer, "QThread": QThread, "QMessageBox": QMessageBox, "QDesktopServices": QDesktopServices,
                "QUrl": QUrl, "QSplitter": QSplitter, "QTabWidget": QTabWidget, "QFileDialog": QFileDialog, "QApplication": QApplication, "Signal": Signal}
    cache = dict(explicit)

    def mk(modname):
        m = types.ModuleType(modname)
        def __getattr__(name):
            if name.startswith("__"): raise AttributeError(name)
            if name not in cache: cache[name] = type(name, (Stub,), {})
            return cache[name]
        m.__getattr__ = __getattr__
        return m
    pkg = types.ModuleType("PySide6")
    for sub in ("QtCore", "QtGui", "QtWidgets"):
        m = mk("PySide6." + sub); setattr(pkg, sub, m); sys.modules["PySide6." + sub] = m
    sys.modules["PySide6"] = pkg
    return explicit
