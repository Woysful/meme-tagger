"""Background threads: image analysis (pipelined) and `ollama pull`."""
from __future__ import annotations
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from ..engine import Engine, EngineConfig, TagOptions, reset_shared
from ..imaging import thumbnail_rgba
from ..vlm import OllamaVLM, VLMError
from .common import pil_to_qimage

THUMB = 160


@dataclass
class Job:
    rid: int
    path: Path
    analyze: bool
    context: dict


_engine: dict = {"cfg": None, "engine": None}


def get_engine(cfg: EngineConfig, store, progress) -> Engine:
    """One engine per configuration; heavy models are shared between engines anyway (engine.shared)."""
    if _engine["engine"] is None or _engine["cfg"] != cfg:
        _engine["engine"], _engine["cfg"] = Engine(cfg, store, progress), cfg
    _engine["engine"].progress = progress
    return _engine["engine"]


def drop_engine():
    _engine["engine"] = _engine["cfg"] = None
    reset_shared()


class AnalysisWorker(QThread):
    """Takes jobs from a queue that can grow while it runs. While the vision model answers for one image, the next
    one is decoded, OCR'd and run through WD14 in a helper thread, so the GPU waits less."""
    row_started = Signal(int)
    row_done = Signal(int, object, QImage)        # row id, Result (None when not analyzed), thumbnail
    row_error = Signal(int, str, QImage)
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, cfg: EngineConfig, opts: TagOptions, store, jobs, engine_factory=get_engine):
        super().__init__()
        self.cfg, self.opts, self.store = cfg, opts, store
        self.engine_factory = engine_factory
        self.engine: Engine | None = None
        self._jobs, self._lock = deque(jobs), threading.Lock()
        self._go = threading.Event(); self._go.set()
        self.needs_engine = any(j.analyze for j in jobs)

    # ---- control (GUI thread) ----
    def add(self, jobs) -> bool:
        """False when the worker cannot take these jobs (analysis requested, but it was started without an engine)."""
        if any(j.analyze for j in jobs) and not self.needs_engine:
            return False
        with self._lock:
            self._jobs.extend(jobs)
        return True

    def drop(self, rids: set[int]):
        with self._lock:
            self._jobs = deque(j for j in self._jobs if j.rid not in rids)

    def remaining(self) -> int:
        with self._lock:
            return len(self._jobs)

    def pause(self):
        self._go.clear()

    def resume(self):
        self._go.set()

    def is_paused(self) -> bool:
        return not self._go.is_set()

    def cancel(self):
        self.requestInterruption(); self._go.set()

    # ---- thread ----
    def _take(self) -> Job | None:
        with self._lock:
            return self._jobs.popleft() if self._jobs else None

    def _prepare(self, job: Job):
        try:
            if job.analyze:
                prep = self.engine.prepare(job.path, self.opts)
                im = prep.image.rgb.copy(); im.thumbnail((THUMB, THUMB))
                return prep, pil_to_qimage(im), None
            return None, pil_to_qimage(thumbnail_rgba(job.path, THUMB)), None
        except Exception as e:
            return None, pil_to_qimage(thumbnail_rgba(job.path, THUMB)), f"{type(e).__name__}: {e}"

    def run(self):
        if self.needs_engine:
            try:
                self.engine = self.engine_factory(self.cfg, self.store, self.status.emit)
                self.engine.warm_up(self.opts.want)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self.failed.emit(msg if isinstance(e, VLMError) else
                                 msg + "\n\nThe local models need internet once to download.")
                return
        pending = None
        with ThreadPoolExecutor(1, thread_name_prefix="prepare") as pool:
            try:
                while not self.isInterruptionRequested():
                    if pending is None:
                        job = self._take()
                        if job is None:
                            break
                        pending = (job, pool.submit(self._prepare, job))
                    while not self._go.wait(0.2):
                        if self.isInterruptionRequested():
                            break
                    if self.isInterruptionRequested():
                        break
                    job, fut = pending
                    nxt = self._take()
                    pending = (nxt, pool.submit(self._prepare, nxt)) if nxt else None
                    self.row_started.emit(job.rid)
                    prep, thumb, err = fut.result()
                    if err:
                        self.row_error.emit(job.rid, err, thumb); continue
                    if prep is None:
                        self.row_done.emit(job.rid, None, thumb); continue
                    try:
                        self.row_done.emit(job.rid, self.engine.analyze(prep, job.context, self.opts), thumb)
                    except Exception as e:
                        self.row_error.emit(job.rid, f"{type(e).__name__}: {e}", thumb)
                    finally:
                        prep.image.release()
            finally:
                if pending is not None:
                    try:
                        prep = pending[1].result()[0]
                        if prep is not None:
                            prep.image.release()
                    except Exception:
                        pass
                if self.engine is not None:
                    self.engine.flush()


class PullWorker(QThread):
    """`ollama pull` with progress; cancelling closes the stream (Ollama resumes next time)."""
    progress = Signal(str, object, object)        # status, downloaded, total (bytes, may be None)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, host, name):
        super().__init__()
        self.host, self.name = host, name

    def cancel(self):
        self.requestInterruption()

    def run(self):
        try:
            OllamaVLM(self.host, self.name).pull(self.name, lambda st, c, t: self.progress.emit(st, c, t),
                                                 self.isInterruptionRequested)
            self.done.emit(self.name)
        except VLMError as e:
            if e.code != "cancel":
                self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
