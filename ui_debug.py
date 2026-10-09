"""Development aid: the grader photographs its own window, without mouse clicks or focus changes.

Idea from the 5-track project (ui_debug.py there uses the Windows API). Here the window content is read
through the X server, which also works for a window that is covered or placed outside the visible screen.
"""


def place_offscreen(window, width=1920, height=1080):
    """Show a Tk window outside the visible screen (no title bar, no focus), so it can be photographed
    without appearing on the operator's display."""
    if window.overrideredirect():               # already placed (the grader does it itself when GSN_UI_OFFSCREEN is set)
        window.update()
        return
    window.withdraw()                           # the window must be unmapped while its type changes
    window.update_idletasks()
    try:
        window.attributes("-zoomed", False)
    except Exception:
        pass
    window.overrideredirect(True)
    window.geometry(f"{width}x{height}+{window.winfo_screenwidth() + 2000}+0")
    window.deiconify()
    window.update()


def capture_window(widget, path) -> bool:
    """Save the top-level window of a Tk widget as PNG. Returns False when it cannot be read."""
    try:
        import gi
        gi.require_version("Gdk", "3.0")
        gi.require_version("GdkX11", "3.0")
        from gi.repository import Gdk, GdkX11
        top = widget.winfo_toplevel()
        top.update_idletasks()
        window = GdkX11.X11Window.foreign_new_for_display(GdkX11.X11Display.get_default(), top.winfo_id())
        pixbuf = Gdk.pixbuf_get_from_window(window, 0, 0, top.winfo_width(), top.winfo_height())
        if pixbuf is None:
            return False
        pixbuf.savev(str(path), "png", [], [])
        return True
    except Exception:
        return False
