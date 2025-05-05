#!/usr/bin/env python3
# pylint: disable=import-error
import os
import gi
# Specify versions before importing gi modules
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gtk, Gdk, GLib, Gst, GstVideo, GstApp
import traceback
import math
import sys
import threading
import yt_dlp
from collections import deque
try:
    import numpy as np
except Exception as e:
    print("numpy import failed:", e)
    traceback.print_exc()
    np = None

try:
    import librosa
except Exception as e:
    print("librosa import failed:", e)
    traceback.print_exc()
    librosa = None

# Optional OpenCV (used for frame capture). Fallback to None if unavailable.
try:
    import cv2  # type: ignore  # pylint: disable=import-error
except Exception as e:
    print("cv2 import failed:", e)
    cv2 = None

print("Starting application...")


class GuitarTrainerApp(Gtk.Window):
    def __init__(self):
        print("Initializing GuitarTrainerApp...")
        super().__init__()
        self.set_title("Guitar Tutorial Player")
        self.set_default_size(800, 600)

        # Cleanup previous debug frame capture if present
        try:
            if os.path.exists("debug_frame.png"):
                os.remove("debug_frame.png")
        except OSError:
            pass

        # Initialize GStreamer
        print("Initializing GStreamer...")
        Gst.init(None)
        self.pipeline = None
        self.playbin = None
        self.videosink = None

        # Create main layout
        self.main_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add(self.main_box)

        # URL entry
        self.url_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.url_entry = Gtk.Entry()
        self.url_entry.set_placeholder_text("Enter YouTube URL")
        self.download_button = Gtk.Button(label="Download")
        self.download_button.connect("clicked", self.on_download_clicked)
        self.url_box.pack_start(self.url_entry, True, True, 0)
        self.url_box.pack_start(self.download_button, False, False, 0)
        self.main_box.pack_start(self.url_box, False, False, 0)

        # Status label
        self.status_label = Gtk.Label(label="")
        self.main_box.pack_start(self.status_label, False, False, 0)

        # Video area
        self.video_area = Gtk.Box()  # Changed from DrawingArea to Box
        self.video_area.set_hexpand(True)
        self.video_area.set_vexpand(True)
        self.main_box.pack_start(self.video_area, True, True, 0)

        # Ensure slider clicks warp directly to the clicked position
        settings = Gtk.Settings.get_default()  # pylint: disable=no-value-for-parameter
        if settings is not None:
            try:
                settings.set_property("gtk-primary-button-warps-slider", True)
            except TypeError:
                # Fallback for older GTK versions where the property might not exist
                pass
        
        # Speed control
        self.speed_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.speed_label = Gtk.Label(label="Playback Speed:")
        self.speed_scale = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, 0.25, 2.0, 0.25)
        self.speed_scale.set_value(1.0)
        self.speed_scale.connect("value-changed", self.on_speed_changed)
        self.speed_box.pack_start(self.speed_label, False, False, 0)
        self.speed_box.pack_start(self.speed_scale, True, True, 0)
        self.main_box.pack_start(self.speed_box, False, False, 0)

        # Playback controls
        self.control_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.play_button = Gtk.Button(label="Play")
        self.play_button.connect("clicked", self.on_play_clicked)
        self.pause_button = Gtk.Button(label="Pause")
        self.pause_button.connect("clicked", self.on_pause_clicked)
        self.control_box.pack_start(self.play_button, False, False, 0)
        self.control_box.pack_start(self.pause_button, False, False, 0)
        self.main_box.pack_start(self.control_box, False, False, 0)

        # Initialize video path
        self.video_path = None
        self.pitch_buffer = None  # numpy array for accumulating audio samples
        self.recent_pitches = deque(maxlen=5)
        # TAB-highlight state
        self.tab_bbox = None        # (x, y, w, h) of tab area in video coords
        self.tab_col = -1           # current column index
        self.col_width = 80         # pixel width per TAB column (heuristic)
        self.highlight_rects = []   # list[(x, y, w, h)] to draw each frame
        self.tab_highlight = None   # overlay widget created later in setup_pipeline
        self.onset_buffer = None    # numpy buffer for fallback onset detection
        self.frame_captured = False  # Flag to track first-frame capture

        # Auto-load first video file in current directory (mp4/avi/mkv/mov)
        for fname in os.listdir('.'):
            if fname.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                self.video_path = fname
                print(f"Found local video file: {self.video_path}")
                GLib.idle_add(self.load_local_video)
                break

        print("GuitarTrainerApp initialization complete")

    def update_status(self, message):
        GLib.idle_add(self.status_label.set_text, message)

    def setup_pipeline(self):
        print("Setting up pipeline...")
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            print("Set existing pipeline to NULL state")

        try:
            # Create pipeline elements
            self.pipeline = Gst.Pipeline.new("pipeline")
            self.playbin = Gst.ElementFactory.make("playbin", "playbin")

            # Create video sink
            self.videosink = Gst.ElementFactory.make("gtkglsink", "videosink")
            if not self.videosink:
                print("gtkglsink not available, trying gtksink")
                self.videosink = Gst.ElementFactory.make(
                    "gtksink", "videosink")

            if self.videosink:
                try:
                    # Make sure video frames are displayed in sync with the pipeline clock
                    self.videosink.set_property("sync", True)
                    print("Set videosink sync property to True")
                except TypeError:
                    print("Could not set sync property on videosink")
                    pass  # some versions may not expose the property

            # Build the audio sink description dynamically so the application
            # still works on systems that do *not* have the aubioonset plugin
            # installed.  We test once and only add the onset branch when the
            # element is available.

            has_aubioonset = Gst.ElementFactory.find("aubioonset") is not None

            base_desc = (
                "audioconvert ! audioresample ! scaletempo name=st ! tee name=split "
                # ──── audible playback (keeps sync=true so speed changes are heard)
                "split. ! queue max-size-time=0 max-size-buffers=0 ! audioconvert ! audioresample ! autoaudiosink sync=true "
                # ──── pitch detector branch
                "split. ! queue ! audioconvert ! audioresample ! capsfilter caps=audio/x-raw,format=F32LE,channels=1 ! "
                "appsink name=pitchsink emit-signals=true sync=false max-buffers=5 drop=true "
            )

            onset_desc = ""
            if has_aubioonset:
                onset_desc = "split. ! queue ! aubioonset name=onsetsink ! fakesink"
            else:
                # Fallback: create an appsink for Python-side onset detection
                print("aubioonset element not found — using librosa onset detection fallback")
                onset_desc = (
                    "split. ! queue ! audioconvert ! audioresample ! "
                    "capsfilter caps=audio/x-raw,format=F32LE,channels=1 ! "
                    "appsink name=onsetsink emit-signals=true sync=false max-buffers=5 drop=true"
                )

            audio_sink_desc = base_desc + onset_desc
            self.audio_sink_bin = Gst.parse_bin_from_description(
                audio_sink_desc, True)

            # Retrieve the appsink for pitch detection
            self.pitchsink = self.audio_sink_bin.get_by_name("pitchsink")
            if self.pitchsink:
                self.pitchsink.connect("new-sample", self.on_pitch_sample)

            # Retrieve the aubioonset element for onset detection so we can
            # identify its messages on the bus later.
            self.onsetsink = self.audio_sink_bin.get_by_name("onsetsink")

            # If the onsetsink is an appsink (fallback), connect signal handler
            if self.onsetsink and isinstance(self.onsetsink, GstApp.AppSink):
                self.onsetsink.connect("new-sample", self.on_onset_sample)

            # Retrieve the scaletempo element and make sure it's set up correctly
            self.scaletempo = self.audio_sink_bin.get_by_name("st")
            if self.scaletempo:
                self.scaletempo.set_property("stride", 30)
                self.scaletempo.set_property("overlap", 0.2)
                print("Configured scaletempo element")

            if not self.playbin or not self.videosink:
                print("Failed to create elements")
                self.update_status("Failed to create video player")
                return

            # Get the widget from the sink and add it to our video area
            sink_widget = self.videosink.get_property("widget")
            if sink_widget:
                # Clear previous content
                for child in self.video_area.get_children():
                    self.video_area.remove(child)

                # Create an overlay to hold video + pitch label
                overlay = Gtk.Overlay()
                overlay.set_hexpand(True)
                overlay.set_vexpand(True)

                # Add video widget first
                overlay.add(sink_widget)

                # Pitch label (top-left)
                self.pitch_label = Gtk.Label(label="")
                self.pitch_label.set_halign(Gtk.Align.START)
                self.pitch_label.set_valign(Gtk.Align.START)
                self.pitch_label.set_margin_start(10)
                self.pitch_label.set_margin_top(10)
                # Style: yellow text for visibility
                self.pitch_label.override_color(
                    Gtk.StateFlags.NORMAL, Gdk.RGBA(1, 1, 0, 1))
                overlay.add_overlay(self.pitch_label)

                # --- Overlay for TAB highlighting ---
                self.tab_highlight = Gtk.DrawingArea()
                self.tab_highlight.set_halign(Gtk.Align.FILL)
                self.tab_highlight.set_valign(Gtk.Align.FILL)
                self.tab_highlight.set_hexpand(True)
                self.tab_highlight.set_vexpand(True)
                self.tab_highlight.connect("draw", self.on_tab_highlight_draw)
                overlay.add_overlay(self.tab_highlight)

                self.video_area.pack_start(overlay, True, True, 0)
                sink_widget.show()
                overlay.show_all()

            # Set up the sinks
            self.playbin.set_property("video-sink", self.videosink)
            self.playbin.set_property("audio-sink", self.audio_sink_bin)

            # Add playbin to pipeline
            self.pipeline.add(self.playbin)

            # Add a message handler
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect('message', self.on_message)

            print("Pipeline setup complete")
        except Exception as e:
            print(f"Error setting up pipeline: {e}")
            self.update_status(f"Error setting up video player: {str(e)}")

    def on_pad_added(self, element, pad):
        print(f"Pad added: {pad.get_name()}")
        pad_type = pad.query_caps(None).to_string()
        print(f"Pad type: {pad_type}")

        if "video" in pad_type:
            pad.link(self.videosink.get_static_pad("sink"))
        elif "audio" in pad_type:
            pad.link(self.audio_sink_bin.get_static_pad("sink"))

    def on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, Debug: {debug}")
            self.update_status(f"Playback error: {err}")
        elif t == Gst.MessageType.EOS:
            print("End of stream")
            self.pipeline.set_state(Gst.State.NULL)
            self.update_status("Playback finished")
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                print(
                    f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}")
                self.update_status(f"Player {new_state.value_nick}")
        elif t == Gst.MessageType.TAG:
            # aubioonset posts TAG messages with an "onset" tag each time it
            # detects a new attack.  We advance the TAB cursor (or simply log
            # for now) when we receive one.
            if message.src == getattr(self, "onsetsink", None):
                # Any TAG message from onsetsink corresponds to an onset.
                GLib.idle_add(self.on_onset_detected)

    def on_download_clicked(self, button):
        url = self.url_entry.get_text()
        if not url:
            self.update_status("Please enter a YouTube URL")
            return

        print(f"Starting download of: {url}")
        self.update_status("Downloading video...")
        self.download_button.set_sensitive(False)

        def download_video():
            try:
                ydl_opts = {
                    'format': 'best[ext=mp4]',
                    'outtmpl': '%(title)s.%(ext)s',
                    'quiet': True,
                    'no_warnings': True
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    self.video_path = f"{info['title']}.mp4"
                    print(f"Download complete: {self.video_path}")
                    GLib.idle_add(self.on_download_complete)
            except Exception as e:
                print(f"Download error: {e}")
                GLib.idle_add(self.update_status, f"Download failed: {str(e)}")
                GLib.idle_add(self.download_button.set_sensitive, True)

        threading.Thread(target=download_video, daemon=True).start()

    def on_download_complete(self):
        print("Download complete, setting up playback...")
        self.update_status("Setting up video player...")
        if self.video_path:
            self.setup_pipeline()
            uri = f"file://{os.path.abspath(self.video_path)}"
            print(f"Setting URI: {uri}")
            self.playbin.set_property("uri", uri)

            # Set initial state to PAUSED to preroll the pipeline
            ret = self.pipeline.set_state(Gst.State.PAUSED)
            print(f"Pipeline state change result: {ret}")

            # Wait for the state change to complete
            ret = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
            print(f"Pipeline get_state result: {ret}")

            # Initialize the first seek now, to ensure the speed is correctly set
            # This addresses the issue of video playback speed
            speed = self.speed_scale.get_value()
            print(f"Setting initial playback speed to {speed}")
            result = self.pipeline.seek(
                speed,  # rate
                Gst.Format.TIME,
                Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE,
                Gst.SeekType.SET,
                0,  # start position
                Gst.SeekType.NONE,
                -1  # we don't specify an end time
            )
            print(f"Initial seek result: {result}")

            if ret[0] == Gst.StateChangeReturn.SUCCESS:
                self.update_status("Video ready to play")
            else:
                self.update_status("Failed to prepare video")
                print(f"Failed to prepare video: {ret}")
        self.download_button.set_sensitive(True)

    def on_play_clicked(self, button):
        if self.playbin:
            self.pitch_buffer = None  # reset analysis buffer
            print("Play button clicked")

            # Set to PLAYING state
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            print(f"Play state change result: {ret}")

            # Wait for the state change to complete
            ret = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
            print(f"Pipeline get_state result after play: {ret}")

            if ret[0] == Gst.StateChangeReturn.SUCCESS:
                self.update_status("Playing")

                # Start trying to capture a non-black frame (once per 0.4 s)
                if not self.frame_captured:
                    GLib.timeout_add(400, self._capture_first_frame)
            else:
                self.update_status("Failed to play video")

    def on_pause_clicked(self, button):
        if self.playbin:
            ret = self.pipeline.set_state(Gst.State.PAUSED)
            print(f"Pause state change result: {ret}")

            # Wait for the state change to complete
            ret = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
            print(f"Pipeline get_state result after pause: {ret}")

            if ret[0] == Gst.StateChangeReturn.SUCCESS:
                self.update_status("Paused")
            else:
                self.update_status("Failed to pause video")

    def on_speed_changed(self, scale):
        speed = scale.get_value()
        if self.pipeline and self.playbin:
            # Query the current playback position so we can seek from there
            success, position = self.pipeline.query_position(Gst.Format.TIME)
            if not success:
                position = 0

            print(
                f"Changing playback speed to {speed}. Current position: {position}")

            # Build a seek event that only changes the playback rate while
            # continuing from the current position. We use FLUSH so the
            # pipeline updates immediately and ACCURATE to improve precision.
            flags = Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE

            # Seek to the current position with new speed
            result = self.pipeline.seek(
                speed,               # new playback rate
                Gst.Format.TIME,
                flags,
                Gst.SeekType.SET,    # start type – absolute position
                position,            # start position (current)
                Gst.SeekType.NONE,   # stop type – play until the end
                -1                    # stop position (ignored when NONE)
            )

            if result:
                print("Seek (rate change) successful")
                self.update_status(f"Playback speed: {speed:.2f}x")
            else:
                print("Seek (rate change) failed")
                self.update_status("Failed to change speed")

    def on_pitch_sample(self, appsink):
        print("on_pitch_sample called")
        global np, librosa
        if np is None:
            try:
                import numpy as np
            except Exception as e:
                print("numpy import failed:", e)
                traceback.print_exc()
                return Gst.FlowReturn.OK
        if librosa is None:
            try:
                import librosa
            except Exception as e:
                print("librosa import failed:", e)
                traceback.print_exc()
                return Gst.FlowReturn.OK

        sample = appsink.emit("pull-sample")
        if sample is None:
            print("sample is None")
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        caps = sample.get_caps()
        rate = caps.get_structure(0).get_value("rate")
        channels = caps.get_structure(0).get_value("channels")

        # Map buffer to numpy array
        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            # audio/x-raw float32 interleaved
            block = np.frombuffer(mapinfo.data, dtype=np.float32)
            if channels > 1:
                block = block.reshape(-1, channels)
                block = block.mean(axis=1)  # mixdown to mono

            # append to rolling buffer
            if self.pitch_buffer is None:
                self.pitch_buffer = block.copy()
            else:
                self.pitch_buffer = np.concatenate((self.pitch_buffer, block))

            FRAME = 2048  # ~46 ms @ 44.1 kHz
            HOP = FRAME // 2  # 50% overlap (~23 ms update)

            # only analyse when we have enough samples
            if len(self.pitch_buffer) < FRAME:
                return Gst.FlowReturn.OK

            data = self.pitch_buffer[:FRAME]
            # keep remainder for next round
            self.pitch_buffer = self.pitch_buffer[HOP:]

            FMIN = librosa.note_to_hz('E2')  # 82 Hz
            FMAX = librosa.note_to_hz('E6')  # 1319 Hz

            f0, voiced_flag, _ = librosa.pyin(data, fmin=FMIN, fmax=FMAX,
                                              sr=rate, frame_length=FRAME,
                                              hop_length=len(data)-1)

            voiced = f0[voiced_flag]
            if voiced.size == 0:
                return Gst.FlowReturn.OK

            freq = float(np.median(voiced))
            self.recent_pitches.append(freq)
            freq_smoothed = float(np.median(self.recent_pitches))

            note = self.freq_to_note_name(freq_smoothed)
            print(f"Detected pitch: {note} {freq_smoothed:.0f}Hz")
            GLib.idle_add(self.pitch_label.set_text,
                          f"{note} ({freq_smoothed:.0f} Hz)")
        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E',
                  'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def freq_to_note_name(self, freq):
        if freq <= 0 or math.isnan(freq):
            return "-"
        note_num = int(round(12 * math.log2(freq / 440.0) + 69))
        octave = note_num // 12 - 1
        name = self.NOTE_NAMES[note_num % 12]
        return f"{name}{octave}"

    def on_onset_detected(self):
        """Callback executed in the GTK main thread when aubioonset signals a new onset.

        At this stage we only print and keep a placeholder for future TAB
        column-advance logic.
        """
        print("Onset detected (aubioonset)")

        # Ensure the highlight overlay exists
        if not self.tab_highlight:
            return

        alloc = self.tab_highlight.get_allocation()
        width, height = alloc.width, alloc.height

        # Lazily determine the TAB bounding box (bottom 35 % of frame)
        if self.tab_bbox is None and width > 0 and height > 0:
            bbox_y = int(height * 0.65)
            bbox_h = int(height * 0.30)
            self.tab_bbox = (0, bbox_y, width, bbox_h)

        if self.tab_bbox is None:
            return

        # Advance column index and wrap when exceeding width
        bbox_x, bbox_y, bbox_w, bbox_h = self.tab_bbox
        self.tab_col = (self.tab_col + 1) % max(1, bbox_w // self.col_width)

        x = bbox_x + self.tab_col * self.col_width
        self.highlight_rects = [(x, bbox_y, self.col_width, bbox_h)]

        print(f"tab_highlight allocation: {width}x{height}")
        print(f"Highlight rects set to: {self.highlight_rects}")

        self.tab_highlight.queue_draw()

    # ---------------------------------------------------------------------
    # Gtk.DrawingArea draw callback for TAB highlight overlay
    # ---------------------------------------------------------------------
    def on_tab_highlight_draw(self, widget, cr):
        """Draw translucent rectangles over the current TAB column.

        This method is connected to the `draw` signal of `self.tab_highlight`.
        It iterates through `self.highlight_rects` and fills each rectangle
        with a semi-transparent colour.
        """
        if not self.highlight_rects:
            # Debug: nothing to draw
            print("on_tab_highlight_draw: no rects")
            return False

        print(f"on_tab_highlight_draw: drawing {len(self.highlight_rects)} rects")

        cr.set_source_rgba(1.0, 0.8, 0.0, 0.35)  # yellow-orange, α=0.35
        for x, y, w, h in self.highlight_rects:
            cr.rectangle(x, y, w, h)
            cr.fill()

        # return False to propagate further drawing if needed
        return False

    # ------------------------------------------------------------------
    # Fallback Python-side onset detection when aubioonset is unavailable
    # ------------------------------------------------------------------
    def on_onset_sample(self, appsink):
        """Signal handler for appsink-based onset detection fallback."""
        global np, librosa
        if np is None:
            try:
                import numpy as np
            except Exception as e:
                print("numpy import failed (onset):", e)
                return Gst.FlowReturn.OK
        if librosa is None:
            try:
                import librosa
            except Exception as e:
                print("librosa import failed (onset):", e)
                return Gst.FlowReturn.OK

        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        caps = sample.get_caps()
        rate = caps.get_structure(0).get_value("rate")

        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            block = np.frombuffer(mapinfo.data, dtype=np.float32)
        finally:
            buf.unmap(mapinfo)

        # Accumulate ~0.1 s of audio before running onset detector
        if self.onset_buffer is None:
            self.onset_buffer = block.copy()
        else:
            self.onset_buffer = np.concatenate((self.onset_buffer, block))

        # Require at least 4096 samples (~93 ms @ 44.1 kHz)
        MIN_SAMPLES = 4096
        if len(self.onset_buffer) < MIN_SAMPLES:
            return Gst.FlowReturn.OK

        # Run onset detection on the accumulated buffer
        try:
            onsets = librosa.onset.onset_detect(y=self.onset_buffer, sr=rate, units="samples", backtrack=False)
        except Exception as e:
            print("librosa onset_detect error:", e)
            onsets = []

        if len(onsets) > 0:
            # Call UI update in main thread
            GLib.idle_add(self.on_onset_detected)
            # Keep residual samples after last onset to continue detection
            last_onset_sample = int(onsets[-1])
            self.onset_buffer = self.onset_buffer[last_onset_sample:]
        else:
            # To prevent the buffer from growing unbounded, keep last slice
            self.onset_buffer = self.onset_buffer[-MIN_SAMPLES:]

        return Gst.FlowReturn.OK

    # --------------------------------------------------------------
    # Load a local video discovered at startup
    # --------------------------------------------------------------
    def load_local_video(self):
        if not self.video_path:
            return

        self.update_status("Preparing local video…")

        # Build pipeline and set URI
        self.setup_pipeline()

        uri = f"file://{os.path.abspath(self.video_path)}"
        print(f"Setting URI for local video: {uri}")
        self.playbin.set_property("uri", uri)

        # Preroll (PAUSED) so first frame is available
        self.pipeline.set_state(Gst.State.PAUSED)
        self.update_status("Local video ready – press Play")

    # --------------------------------------------------------------
    # Poll videosink.last-sample until we capture a bright frame
    # --------------------------------------------------------------
    def _capture_first_frame(self):
        global cv2, np  # pylint: disable=undefined-variable

        # Lazy import cv2 / numpy if they were not available at startup
        if cv2 is None:
            try:
                import cv2 as _cv2  # type: ignore  # pylint: disable=import-error
                globals()['cv2'] = _cv2
            except Exception as e:
                print('[DEBUG] cv2 import failed in capture:', e)
                return True  # keep trying (maybe package loads later)

        if np is None:
            try:
                import numpy as _np  # type: ignore
                globals()['np'] = _np
            except Exception as e:
                print('[DEBUG] numpy import failed in capture:', e)
                return False

        if self.frame_captured or self.videosink is None:
            return False  # stop timer

        sample = self.videosink.get_property("last-sample")
        if not sample:
            return True  # try again later

        buf = sample.get_buffer()
        caps = sample.get_caps()
        w = caps.get_structure(0).get_value("width")
        h_caps = caps.get_structure(0).get_value("height")

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return False

        try:
            # Infer bytes-per-pixel from buffer size.  This handles RGB, BGRx, etc.
            buf_size = len(mapinfo.data)
            if w == 0:
                return False  # invalid caps

            bpp = buf_size // (w * h_caps) if h_caps > 0 else 0

            if bpp not in (3, 4):
                print(f"[DEBUG] Unsupported bytes-per-pixel ({bpp}), skipping frame capture")
                return False

            # Some sinks misreport height in caps (e.g. 360 vs actual 480).
            # Re-compute real height from buffer size.
            h_real = buf_size // (w * bpp)

            if h_real * w * bpp != buf_size:
                print("[DEBUG] Buffer size does not align with inferred geometry, skipping")
                return False

            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h_real, w, bpp))

            # Convert to 3-channel BGR for saving if necessary
            if bpp == 4:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:  # bpp == 3, assume BGR already (gtksink outputs BGR)
                frame_bgr = frame

            if frame_bgr.mean() < 5:
                return True  # keep polling until non-black

            cv2.imwrite("debug_frame.png", frame_bgr)

            # Attempt to locate TAB area
            bbox = self.detect_tab_bbox(frame_bgr)
            if bbox:
                print(f"[DEBUG] TAB bbox detected: {bbox}")
                self.tab_bbox = bbox
                # Draw a rectangle in debug image for visual verification
                dbg = frame_bgr.copy()
                x, y, w_box, h_box = bbox
                cv2.rectangle(dbg, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.imwrite("tab_detect.png", dbg)

                # Update highlight overlay in GTK thread
                if self.tab_highlight:
                    self.highlight_rects = [bbox]
                    GLib.idle_add(self.tab_highlight.queue_draw)

            print(f"[DEBUG] First frame captured ({w}x{h_real}, bpp={bpp}) -> debug_frame.png")
            self.frame_captured = True
            self.update_status("Captured frame for TAB detection")
        finally:
            buf.unmap(mapinfo)

        return False  # stop polling once captured

    # --------------------------------------------------------------
    # Detect TAB bounding box (group of 6 horizontal lines) in frame
    # --------------------------------------------------------------
    def detect_tab_bbox(self, img_bgr):
        """Return (x, y, w, h) bounding box of guitar TAB if found, else None."""
        if cv2 is None or np is None:
            return None

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        roi_y0 = int(h * 0.45)
        roi = gray[roi_y0:, :]

        # Crop away black side bars by checking column brightness
        col_sum = np.mean(roi, axis=0)
        non_dark = np.where(col_sum > 10)[0]  # columns with some brightness
        if len(non_dark) < w * 0.3:  # screen mostly black? give up
            print("[DEBUG] ROI mostly dark") 
            return None

        x_left, x_right = non_dark[0], non_dark[-1]
        roi = roi[:, x_left:x_right]
        w_cropped = x_right - x_left

        # Adaptive threshold → binary with lines in white
        bin_img = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 15, 9)

        # Morphological opening to keep only horizontal lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        lines_only = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)

        # Row-wise projection
        proj = np.sum(lines_only, axis=1)
        if proj.max() == 0:
            print("[DEBUG] Projection all zeros – no lines")
            return None

        thresh = 0.3 * proj.max()
        peak_rows = np.where(proj >= thresh)[0]

        if len(peak_rows) < 6:
            print(f"[DEBUG] Only {len(peak_rows)} peak rows; need >=6")
            return None

        # Cluster consecutive rows into individual line centers
        clusters = []
        current = [peak_rows[0]]
        for r in peak_rows[1:]:
            if r - current[-1] <= 2:
                current.append(r)
            else:
                clusters.append(current)
                current = [r]
        clusters.append(current)

        centers = [int(np.mean(c)) for c in clusters]
        centers.sort()

        # Find any 6-line subset with roughly equal spacing (±3 px)
        best_group = None
        best_score = 9999
        for i in range(len(centers) - 5):
            group = centers[i:i + 6]
            spacings = np.diff(group)
            if spacings.min() == 0:
                continue
            if np.max(spacings) - np.min(spacings) <= 3:
                score = np.std(spacings)
                if score < best_score:
                    best_score = score
                    best_group = group

        if best_group is None:
            print("[DEBUG] No 6-line uniform group from projection; falling back to Hough")
            # ‑-- Fallback: call previous Hough-based method for robustness
            return self.detect_tab_bbox_hough(img_bgr)

        mean_sp = int(np.mean(np.diff(best_group)))
        y_top = best_group[0] - mean_sp // 2 + roi_y0
        y_bot = best_group[-1] + mean_sp // 2 + roi_y0
        bbox_h = y_bot - y_top
        bbox_x = x_left
        return (bbox_x, max(y_top, 0), w_cropped, bbox_h)

    # ------------------------------------------------------------------
    # Previous Hough-based detector kept for fallback / debugging
    # ------------------------------------------------------------------
    def detect_tab_bbox_hough(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape
        roi_y0 = int(h * 0.45)
        roi = gray[roi_y0:, :]
        roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
        edges = cv2.Canny(roi_blur, 50, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                                minLineLength=int(w * 0.4), maxLineGap=30)
        if lines is None:
            print("[DEBUG] Hough fallback also found no lines")
            return None

        ys = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if abs(y1 - y2) <= 4:
                ys.append(y1)
        if len(ys) < 6:
            print(f"[DEBUG] Hough fallback found {len(ys)} lines < 6")
            return None

        ys = np.array(sorted(ys))
        clusters = []
        current = [ys[0]]
        for y in ys[1:]:
            if y - current[-1] <= 6:
                current.append(y)
            else:
                clusters.append(current)
                current = [y]
        clusters.append(current)
        centers = [int(np.mean(c)) for c in clusters]
        centers.sort()
        for i in range(len(centers) - 5):
            group = centers[i:i + 6]
            spacings = np.diff(group)
            if np.max(spacings) - np.min(spacings) <= 4:
                mean_sp = int(np.mean(spacings))
                y_top = group[0] - mean_sp // 2 + roi_y0
                y_bot = group[-1] + mean_sp // 2 + roi_y0
                return (0, y_top, w, y_bot - y_top)
        print("[DEBUG] Hough fallback no uniform 6-line group")
        return None


if __name__ == '__main__':
    print("Starting main...")
    win = GuitarTrainerApp()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    print("Running application...")
    Gtk.main()
    print("Application finished")
