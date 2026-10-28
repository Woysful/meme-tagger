"""Контраст интерфейса по WCAG: защита от «элементы сливаются с фоном»."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memetagger.theme import THEMES, blend, build_style, contrast, make_icons


def test_contrast():
    for name, t in THEMES.items():
        c = lambda a, b: contrast(t[a], t[b])
        for bg in ("window", "card", "raised", "input", "hover", "alt"):
            assert c("text", bg) >= 7, (name, "text", bg, c("text", bg))
        for bg in ("window", "card", "raised", "input"):
            assert c("muted", bg) >= 4.5, (name, "muted", bg, c("muted", bg))
        for bg in ("accent", "accent_hover"):
            assert c("on_accent", bg) >= 4.5, (name, "on_accent", bg, c("on_accent", bg))
        for fg in ("ok", "bad"):
            for bg in ("card", "raised"):
                assert c(fg, bg) >= 4.5, (name, fg, bg, c(fg, bg))
        for bg in ("card", "raised"):
            assert c("disabled", bg) >= 3, (name, "disabled", bg)
        for bg in ("card", "window", "input"):
            assert c("border", bg) >= 2.5, (name, "border", bg, c("border", bg))           # рамки/границы видны
        assert c("border", "raised") >= 2.3, (name, "border on raised")
        assert c("accent", "card") >= 2.8 and c("accent", "window") >= 2.8                  # акцент отличим от фона
        for bg in ("card", "alt"):
            assert contrast(t["text"], blend(t["sel"], t[bg])) >= 4.5, (name, "selection")
        assert len({t["window"], t["card"], t["raised"], t["input"]}) == 4                  # слои различимы по цвету
        assert c("handle", "card") >= 2.8, (name, "scrollbar handle")


def test_stylesheet_is_self_contained(tmp_path):
    for name, t in THEMES.items():
        icons = make_icons(tmp_path, t)
        assert all(icons.values()) and all(Path(p).exists() for p in icons.values())
        css = build_style(t, {**icons, "check": "/tmp/check.png"})
        assert "palette(" not in css, "стили не должны зависеть от системных ролей палитры"
        assert "{t[" not in css and "{{" not in css
        for sel in ("QTabBar::tab:selected", "QTabBar::tab {", "QCheckBox::indicator:checked", "QMenu::item:selected", "QToolTip",
                    'QLabel#badge[state="ok"]', 'QLabel#hint[state="warn"]', "QPushButton#primary"):
            assert sel in css, sel
        assert 'url("/tmp/check.png")' in css and f'url("{icons["down"]}")' in css
        assert "QComboBox::drop-down" in css and "border-top-right-radius" in css
        assert css.count("{") == css.count("}")


if __name__ == "__main__":
    import tempfile
    test_contrast(); test_stylesheet_is_self_contained(Path(tempfile.mkdtemp())); print("OK theme")
