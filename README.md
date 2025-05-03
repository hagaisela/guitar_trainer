# Guitar Trainer

A simple GTK + GStreamer Python application that lets you download a YouTube video and practice it at different speeds **without changing pitch**.

## Features

* Download YouTube videos via `yt-dlp`.
* Playback with video and audio using GStreamer `playbin`.
* Change playback speed from 0.25× to 2×; the built-in `scaletempo` filter preserves the original pitch.
* Minimal GTK UI: URL entry, status label, speed slider, play/pause buttons.

## Requirements

* Python 3.8+
* GStreamer 1.20+ with the `good`, `bad` and `ugly` plugin sets (for `scaletempo`, `gtksink` or `gtkglsink`).
* `pygobject` (GTK3 bindings)
* [`yt-dlp`](https://github.com/yt-dlp/yt-dlp)

### macOS (Homebrew example)

```bash
brew install python3 pygobject3 gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly yt-dlp
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pygobject yt-dlp
```

### Debian / Ubuntu

```bash
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0 \
                 gstreamer1.0-gtk3 gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad,ugly} \
                 yt-dlp python3-pip
pip3 install --user yt-dlp
```

## Running

```bash
python3 guitar_trainer.py
```
Enter a YouTube URL, click **Download**, then use the slider to slow down or speed up the video while practising.

## Development

Linter warnings are shown for GTK objects because static analyzers don't fully understand PyGObject dynamic properties. They're safe to ignore.

---
Focus on practising—happy playing! 🎸 