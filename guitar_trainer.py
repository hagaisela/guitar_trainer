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
import time
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

try:
    from tensorflow.keras.models import load_model  # type: ignore  # pylint: disable=import-error
except Exception as e:
    print("TensorFlow/Keras import failed:", e)
    load_model = None

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
        self.pitch_appsink = None  # For pitch detection
        self.onset_appsink = None  # For onset detection
        self.pitch_buffer = None  # For accumulating audio samples
        self.onset_buffer = None  # numpy buffer for fallback onset detection
        self.recent_pitches = deque(maxlen=5)  # Keep last 5 pitches for smoothing
        self.last_position = 0  # Track last playback position
        self.is_playing = False  # Track play state
        self.pitch_sampling_interval = 0.1  # 100ms between regular pitch samples
        self.last_pitch_time = 0  # For regular pitch sampling
        self.last_onset_time = 0  # For onset detection

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

        # Pitch label
        self.pitch_label = Gtk.Label(label="")
        self.pitch_label.set_markup("<span size='x-large'>-</span>")
        self.main_box.pack_start(self.pitch_label, False, False, 0)

        # Video area
        self.video_area = Gtk.Box()  # Changed from DrawingArea to Box
        self.video_area.set_hexpand(True)
        self.video_area.set_vexpand(True)
        self.main_box.pack_start(self.video_area, True, True, 0)

        # ------------------------------ NEW: Seek / Scrub bar ------------------------------
        self.user_is_seeking = False  # Track if the user is currently dragging the bar
        self.seek_scale = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, 0.0, 100.0, 1.0)  # range reset once video is loaded
        self.seek_scale.set_hexpand(True)
        self.seek_scale.set_draw_value(True)  # show formatted time
        self.seek_scale.set_sensitive(False)  # becomes active once duration is known
        self.seek_scale.connect("format-value", self._format_time)
        # Receive mouse button events so we know when the user starts/ends dragging
        self.seek_scale.add_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK)
        self.seek_scale.connect("button-press-event", self._on_seek_button_press)
        self.seek_scale.connect("button-release-event", self._on_seek_button_release)
        self.main_box.pack_start(self.seek_scale, False, True, 0)
        # -------------------------------------------------------------------------------

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
        # Change to continuous scale with finer control
        self.speed_scale = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, 0.1, 2.0, 0.01)  # 0.1x to 2.0x with 0.01 steps
        self.speed_scale.set_value(1.0)
        self.speed_scale.set_digits(2)  # Show 2 decimal places
        self.speed_scale.set_draw_value(True)  # Show the value
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
        self.tab_bbox = None        # (x, y, w, h) of tab area in video coords
        self.tab_col = -1           # current column index
        self.col_width = 80         # pixel width per TAB column (heuristic)
        self.highlight_rects = []   # list[(x, y, w, h)] to draw each frame
        self.tab_highlight = None   # overlay widget created later in setup_pipeline
        self.frame_captured = False  # Flag to track first-frame capture
        # ------------------------------------------------------------------
        # Digit-recognition CNN (trained via train_tab_digit_model.py)
        # ------------------------------------------------------------------
        self.digit_model = self._load_digit_model()

        # Auto-load first video file in current directory (mp4/avi/mkv/mov)
        for fname in os.listdir('.'):
            if fname.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                self.video_path = fname
                print(f"Found local video file: {self.video_path}")
                GLib.idle_add(self.load_local_video)
                break

        self.scaletempo_kicked = False  # Track if scaletempo has been kicked for this video

        # Periodically refresh the seek bar (every 500 ms)
        GLib.timeout_add(500, self._update_seek_bar)

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
                self.videosink = Gst.ElementFactory.make("gtksink", "videosink")

            # Create audio sink bin with tee for branching
            audio_sink_desc = (
                "tee name=audio_tee ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! "
                "audioconvert ! audioresample ! scaletempo ! autoaudiosink sync=false "
                "audio_tee. ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! "
                "audioconvert ! audioresample ! capsfilter caps=audio/x-raw,format=F32LE,channels=1 ! "
                "appsink name=pitch_appsink sync=false max-buffers=1 drop=true "
                "audio_tee. ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! "
                "audioconvert ! audioresample ! capsfilter caps=audio/x-raw,format=F32LE,channels=1 ! "
                "appsink name=onset_appsink sync=false max-buffers=1 drop=true"
            )
            print(f"Creating audio sink bin with description: {audio_sink_desc}")
            self.audio_sink_bin = Gst.parse_bin_from_description(audio_sink_desc, True)
            if not self.audio_sink_bin:
                print("Failed to create audio sink bin")
                return

            # Get the appsinks from the bin
            self.pitch_appsink = self.audio_sink_bin.get_by_name("pitch_appsink")
            self.onset_appsink = self.audio_sink_bin.get_by_name("onset_appsink")
            if not self.pitch_appsink or not self.onset_appsink:
                print("Failed to get appsinks from bin")
                return
            print("Got appsinks from bin")
            self.pitch_appsink.set_property("emit-signals", True)
            self.pitch_appsink.connect("new-sample", self.on_pitch_sample)
            self.onset_appsink.set_property("emit-signals", True)
            self.onset_appsink.connect("new-sample", self.on_onset_sample)

            if not self.playbin or not self.videosink or not self.audio_sink_bin:
                print("Failed to create elements")
                self.update_status("Failed to create video player")
                return

            # Get the widget from the sink and add it to our video area
            sink_widget = self.videosink.get_property("widget")
            if sink_widget:
                # Clear previous content
                for child in self.video_area.get_children():
                    self.video_area.remove(child)

                # Add video widget directly to video area
                self.video_area.pack_start(sink_widget, True, True, 0)
                sink_widget.show()

            # Set up the sinks
            print("Setting up playbin sinks")
            self.playbin.set_property("video-sink", self.videosink)
            self.playbin.set_property("audio-sink", self.audio_sink_bin)
            self.playbin.set_property("flags", 0x00000001 | 0x00000002)  # VIDEO | AUDIO

            # Add playbin to pipeline
            self.pipeline.add(self.playbin)

            # Add a message handler
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect('message', self.on_message)

            print("Pipeline setup complete")

            # Ask the sink for the negotiated video size and resize the window
            caps = self.videosink.get_static_pad("sink").get_current_caps()
            if caps:
                w = caps.get_structure(0).get_value("width")
                h = caps.get_structure(0).get_value("height")
                if w and h:
                    # grow the window to the video size (+ UI chrome margins)
                    GLib.idle_add(self.resize, w, h + 140)
        except Exception as e:
            print(f"Error setting up pipeline: {e}")
            traceback.print_exc()  # Print full stack trace
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
            # Reset state on error
            self.is_playing = False
            self.pipeline.set_state(Gst.State.NULL)
        elif t == Gst.MessageType.EOS:
            print("End of stream")
            self.pipeline.set_state(Gst.State.NULL)
            self.is_playing = False
            self.last_position = 0
            self.update_status("Playback finished")
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                print(f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}")
                if new_state == Gst.State.PLAYING:
                    print("Pipeline is now playing - audio should start flowing")
                    self.is_playing = True
                elif new_state == Gst.State.PAUSED:
                    self.is_playing = False
                elif new_state == Gst.State.NULL:
                    self.is_playing = False
                self.update_status(f"Player {new_state.value_nick}")
        elif t == Gst.MessageType.STREAM_STATUS:
            status = message.parse_stream_status()
            print(f"Stream status: {status.type.value_nick}")
        elif t == Gst.MessageType.NEW_CLOCK:
            print("New clock selected")
        elif t == Gst.MessageType.STREAM_START:
            print("Stream started")
        elif t == Gst.MessageType.ASYNC_DONE:
            print("Async done")
            # Update position after async operations
            if self.is_playing:
                success, position = self.pipeline.query_position(Gst.Format.TIME)
                if success:
                    self.last_position = position
                    print(f"Updated position: {position}")
        elif t == Gst.MessageType.TAG:
            tag_list = message.parse_tag()
            print(f"Tags: {tag_list.to_string()}")
        elif t == Gst.MessageType.BUFFERING:
            percent = message.parse_buffering()
            print(f"Buffering: {percent}%")
        elif t == Gst.MessageType.DURATION_CHANGED:
            print("Duration changed")
        elif t == Gst.MessageType.LATENCY:
            print("Latency message")
        elif t == Gst.MessageType.STEP_DONE:
            print("Step done")
        elif t == Gst.MessageType.ELEMENT:
            print(f"Element message: {message.src.get_name()}")
        elif t == Gst.MessageType.SEGMENT_START:
            print("Segment start")
        elif t == Gst.MessageType.SEGMENT_DONE:
            print("Segment done")
        elif t == Gst.MessageType.TOC:
            print("TOC message")
        elif t == Gst.MessageType.QOS:
            print("QOS message")
        elif t == Gst.MessageType.NEED_CONTEXT:
            print("Need context")
        elif t == Gst.MessageType.HAVE_CONTEXT:
            print("Have context")
        else:
            print(f"Unhandled message type: {t.value_nick}")

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
                    # Fetch highest-resolution video + best audio, then merge into MP4.
                    # This avoids the 720 p cap of progressive MP4 and keeps TAB digits sharp.
                    'format': 'bestvideo+bestaudio/best',
                    'merge_output_format': 'mp4',
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
            ret = self.pipeline.set_state(Gst.State.PAUSED)
            print(f"Pipeline state change result: {ret}")
            ret = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
            print(f"Pipeline get_state result: {ret}")

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

            if self.is_playing:
                print("Already playing, ignoring play request")
                return

            # If we have a saved position, seek to it first
            if self.last_position > 0:
                print(f"Seeking to saved position: {self.last_position}")
                flags = Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE
                result = self.pipeline.seek(
                    1.0,  # rate
                    Gst.Format.TIME,
                    flags,
                    Gst.SeekType.SET,
                    self.last_position,
                    Gst.SeekType.NONE,
                    -1
                )
                if not result:
                    print("Failed to seek to saved position")

            ret = self.pipeline.set_state(Gst.State.PLAYING)
            print(f"Play state change result: {ret}")

            state_change = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
            print(f"Pipeline get_state result after play: {state_change}")

            if state_change[0] == Gst.StateChangeReturn.SUCCESS:
                self.update_status("Playing")
                self.is_playing = True
                if not self.frame_captured:
                    GLib.timeout_add(400, self._capture_first_frame)
            else:
                self.update_status("Failed to play video")
                print(f"Failed to play: {state_change}")

    def on_pause_clicked(self, button):
        if self.playbin:
            # Save current position before pausing
            success, position = self.pipeline.query_position(Gst.Format.TIME)
            if success:
                self.last_position = position
                print(f"Saved position: {position}")

            ret = self.pipeline.set_state(Gst.State.PAUSED)
            print(f"Pause state change result: {ret}")

            # Wait for the state change to complete
            ret = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
            print(f"Pipeline get_state result after pause: {ret}")

            if ret[0] == Gst.StateChangeReturn.SUCCESS:
                self.update_status("Paused")
                self.is_playing = False
            else:
                self.update_status("Failed to pause video")
                print(f"Failed to pause: {ret}")

    def on_speed_changed(self, scale):
        speed = scale.get_value()
        if not self.pipeline or not self.is_playing:
            return

        print(f"Changing playback speed to {speed:.2f}")
        
        # Get current position
        success, position = self.pipeline.query_position(Gst.Format.TIME)
        if not success:
            print("Failed to query position")
            return

        # Build seek event with new rate
        flags = (Gst.SeekFlags.FLUSH | 
                Gst.SeekFlags.ACCURATE | 
                Gst.SeekFlags.KEY_UNIT)  # Add KEY_UNIT for better seeking

        # Calculate stop position (end of stream)
        success, duration = self.pipeline.query_duration(Gst.Format.TIME)
        if not success:
            duration = -1

        # Seek with new rate
        result = self.pipeline.seek(
            speed,               # rate
            Gst.Format.TIME,
            flags,
            Gst.SeekType.SET,    # start type
            position,            # start position
            Gst.SeekType.SET,    # stop type
            duration            # stop position
        )

        if result:
            print(f"Speed change successful: {speed:.2f}x")
            self.update_status(f"Playback speed: {speed:.2f}x")
        else:
            print("Speed change failed")
            self.update_status("Failed to change speed")

    def on_pitch_sample(self, appsink):
        # print("on_pitch_sample called")
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

        print(f"Got audio sample: rate={rate}, channels={channels}")

        # Map buffer to numpy array
        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            print("Failed to map buffer")
            return Gst.FlowReturn.ERROR

        try:
            # audio/x-raw float32 interleaved — mapinfo.data may be bytes,
            # memoryview *or* (with newer GI bindings) a list[int].  Ensure
            # we always hand a real bytes-like buffer to NumPy.
            buf_data = mapinfo.data
            if not isinstance(buf_data, (bytes, bytearray, memoryview)):
                buf_data = bytes(buf_data)

            block = np.frombuffer(buf_data, dtype=np.float32)
            if channels > 1:
                block = block.reshape(-1, channels)
                block = block.mean(axis=1)  # mixdown to mono

            print(f"Got audio block: shape={block.shape}, dtype={block.dtype}")

            # append to rolling buffer
            if self.pitch_buffer is None:
                self.pitch_buffer = block.copy()
            else:
                self.pitch_buffer = np.concatenate((self.pitch_buffer, block))

            FRAME = 2048  # ~46 ms @ 44.1 kHz
            HOP = FRAME // 2  # 50% overlap (~23 ms update)

            # only analyse when we have enough samples
            if len(self.pitch_buffer) < FRAME:
                print(f"Not enough samples yet: {len(self.pitch_buffer)} < {FRAME}")
                return Gst.FlowReturn.OK

            data = self.pitch_buffer[:FRAME]
            # keep remainder for next round
            self.pitch_buffer = self.pitch_buffer[HOP:]

            FMIN = librosa.note_to_hz('E2')  # 82 Hz
            FMAX = librosa.note_to_hz('E6')  # 1319 Hz

            print(f"Running pitch detection on {len(data)} samples")
            
            # Check if this is a regular sampling or onset-triggered
            current_time = time.time()
            is_regular_sample = (current_time - self.last_pitch_time) >= self.pitch_sampling_interval
            
            # Only run pitch detection if:
            # 1. This is a regular sample (every 100ms)
            # 2. OR this is triggered by an onset (handled in on_onset_detected)
            if not is_regular_sample and not hasattr(self, '_onset_triggered'):
                return Gst.FlowReturn.OK
                
            # Clear onset trigger flag if it was set
            if hasattr(self, '_onset_triggered'):
                delattr(self, '_onset_triggered')
                
            # Update last pitch time for regular sampling
            if is_regular_sample:
                self.last_pitch_time = current_time

            f0, voiced_flag, _ = librosa.pyin(data, fmin=FMIN, fmax=FMAX,
                                            sr=rate, frame_length=FRAME,
                                            hop_length=len(data)-1)

            voiced = f0[voiced_flag]
            if voiced.size == 0:
                print("No voiced frames detected")
                return Gst.FlowReturn.OK

            freq = float(np.median(voiced))
            self.recent_pitches.append(freq)
            freq_smoothed = float(np.median(self.recent_pitches))

            note = self.freq_to_note_name(freq_smoothed)
            print(f"Detected pitch: {note} ({freq_smoothed:.0f} Hz)")
            GLib.idle_add(self.pitch_label.set_markup,
                          f"<span size='x-large'>{note} ({freq_smoothed:.0f} Hz)</span>")
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
        """Called when an onset is detected, triggers immediate pitch detection."""
        print("Onset detected, triggering pitch detection")
        self._onset_triggered = True  # This will trigger pitch detection in the next sample
        self.last_onset_time = time.time()

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
            # audio/x-raw float32 interleaved — mapinfo.data may be bytes,
            # memoryview *or* (with newer GI bindings) a list[int].  Ensure
            # we always hand a real bytes-like buffer to NumPy.
            buf_data = mapinfo.data
            if not isinstance(buf_data, (bytes, bytearray, memoryview)):
                buf_data = bytes(buf_data)

            block = np.frombuffer(buf_data, dtype=np.float32)
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

                # Optionally detect and annotate digits within TAB area
                if self.digit_model is not None:
                    self.detect_tab_digits(frame_bgr, bbox)

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
        # Use a full string spacing as padding above the top line and below the
        # bottom line so that the numbers that usually sit slightly outside the
        # six staff lines are fully captured.  A half-spacing (previous logic)
        # was cropping them off.
        padding = mean_sp  # one string spacing

        y_top = best_group[0] - padding + roi_y0
        y_bot = best_group[-1] + padding + roi_y0

        # Clamp to image bounds
        y_top = max(y_top, 0)
        y_bot = min(y_bot, h)

        bbox_h = y_bot - y_top
        bbox_x = x_left
        return (bbox_x, y_top, w_cropped, bbox_h)

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

                # Apply full string spacing padding to include digits outside
                # the six lines.  This prevents cropping of numbers above or
                # below the TAB staff.
                padding = mean_sp  # one string spacing

                y_top = group[0] - padding + roi_y0
                y_bot = group[-1] + padding + roi_y0

                y_top = max(y_top, 0)
                y_bot = min(y_bot, h)

                return (0, y_top, w, y_bot - y_top)
        print("[DEBUG] Hough fallback no uniform 6-line group")
        return None

    # ------------------------------------------------------------------
    # New: digit recognition within detected TAB bbox
    # ------------------------------------------------------------------
    def _load_digit_model(self):
        """Attempt to load the CNN trained by train_tab_digit_model.py.

        Searches a few common locations and returns the loaded Keras model or
        None if loading failed / TensorFlow unavailable.
        """
        if load_model is None:
            return None

        candidate_paths = [
            "tab_digit_cnn.h5",
            os.path.join("models", "tab_digit_cnn.h5"),
            "best.h5",
            os.path.join("models", "best.h5"),
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                try:
                    print(f"Loading TAB-digit CNN from {path}…")
                    model = load_model(path, compile=False)
                    return model
                except Exception as exc:  # pragma: no cover
                    print("[WARN] Failed to load", path, str(exc))
        print("[WARN] No TAB-digit model found — digit detection disabled")
        return None

    def _prep_digit_crop(self, crop_gray_inv):  # -> np.ndarray shape (1,40,40,1)
        """Return crop resized/padded to 40×40 with pixel range [0,1]."""
        import numpy as _np  # local import to avoid top-level requirement
        h_c, w_c = crop_gray_inv.shape
        size = max(h_c, w_c)
        # Use white (255) for the padding background so that after the final
        # inversion step the padding becomes black.  This avoids the bright
        # border lines that were visible in the 40×40 debug crops.
        padded = _np.full((size, size), 255, dtype=_np.uint8)
        y_off = (size - h_c) // 2
        x_off = (size - w_c) // 2
        padded[y_off:y_off + h_c, x_off:x_off + w_c] = crop_gray_inv

        # Thicken strokes a bit (matches MaxFilter augmentation during training)
        # padded = cv2.dilate(padded, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)  # removed dilation to match training preprocessing

        resized = cv2.resize(padded, (40, 40), interpolation=cv2.INTER_NEAREST)

        # Normalise → 0-1 and invert so digits are bright (1) on black (0),
        # exactly like the training generator (arr = 1-arr)
        arr = resized.astype(_np.float32) / 255.0  # keep digits white (1) on black (0)
        arr = 1.0 - arr
        arr = _np.expand_dims(arr, axis=-1)  # HWC→HWC1
        arr = _np.expand_dims(arr, axis=0)   # batch dim
        return arr

    def detect_tab_digits(self, img_bgr, bbox):
        """Detect and annotate fret numbers inside *bbox* of *img_bgr*.

        The routine filters out the vertical "TAB" lettering that sits in the
        extreme left margin and only accepts CNN predictions whose confidence
        is ≥ 0.65.  Annotated boxes are saved to *tab_boxes.png* for debugging.
        """
        if self.digit_model is None or cv2 is None or np is None:
            return

        x0, y0, w_box, h_box = bbox
        roi_bgr = img_bgr[y0 : y0 + h_box, x0 : x0 + w_box]
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Threshold so digits are bright (255) on black (0)
        _, thresh_inv = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Remove the thin staff lines
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        lines_removed = cv2.morphologyEx(
            thresh_inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1
        )
        digits_mask = cv2.bitwise_xor(thresh_inv, lines_removed)

        # ------------------------------------------------------------------
        # 1. Find candidate blobs (contours)
        # ------------------------------------------------------------------
        boxes = []  # each item → [x, y, w, h]
        contours, _ = cv2.findContours(
            digits_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < 8 or w < 4:  # very tiny – noise
                continue
            if h > h_box * 0.9:  # way too tall – probably a bar-line artefact
                continue
            boxes.append([x, y, w, h])

        # ------------------------------------------------------------------
        # 2. Drop anything that lives in the extreme left margin (vertical "TAB")
        # ------------------------------------------------------------------
        roi_w = digits_mask.shape[1]
        LEFT_MARGIN_FRAC = 0.08  # ignore left-most 8 % of the ROI
        boxes = [b for b in boxes if (b[0] + b[2]) > LEFT_MARGIN_FRAC * roi_w]

        # ------------------------------------------------------------------
        # 3. Sort left → right and merge blobs separated by ≤ 2 px
        # ------------------------------------------------------------------
        boxes.sort(key=lambda b: b[0])
        GAP_PX = 2
        MIN_W = 4
        merged = []  # [x0, y0, x1, y1]
        for x, y, w, h in boxes:
            if w < MIN_W:
                continue
            x1, y1 = x + w, y + h
            if merged and x - merged[-1][2] <= GAP_PX:
                # Extend previous box
                merged[-1][2] = max(merged[-1][2], x1)
                merged[-1][1] = min(merged[-1][1], y)
                merged[-1][3] = max(merged[-1][3], y1)
            else:
                merged.append([x, y, x1, y1])

        # ------------------------------------------------------------------
        # 4. Split unusually wide boxes (likely two digits stuck together)
        # ------------------------------------------------------------------
        split_boxes = []
        for x0b, y0b, x1b, y1b in merged:
            w_b = x1b - x0b
            h_b = y1b - y0b
            if w_b > 1.4 * h_b:
                col_sum = digits_mask[y0b:y1b, x0b:x1b].sum(axis=0)
                mid_lo = int(w_b * 0.3)
                mid_hi = int(w_b * 0.7)
                if mid_hi > mid_lo:
                    split_idx = x0b + mid_lo + int(np.argmin(col_sum[mid_lo:mid_hi]))
                    split_boxes.append((x0b, y0b, split_idx, y1b))
                    split_boxes.append((split_idx, y0b, x1b, y1b))
                    continue
            split_boxes.append((x0b, y0b, x1b, y1b))

        # ------------------------------------------------------------------
        # 5. Classify each cropped blob with the CNN
        # ------------------------------------------------------------------
        annotated = roi_bgr.copy()
        PAD = 4
        CONF_THRESH = 0.65  # << raised from 0.30 → 0.65
        for i, (x0b, y0b, x1b, y1b) in enumerate(split_boxes):
            # Pad crop slightly for safety
            x0e = max(0, x0b - PAD)
            y0e = max(0, y0b - PAD)
            x1e = min(digits_mask.shape[1], x1b + PAD)
            y1e = min(digits_mask.shape[0], y1b + PAD)

            crop_mask = digits_mask[y0e:y1e, x0e:x1e]
            raw_crop = thresh_inv[y0e:y1e, x0e:x1e]
            cv2.imwrite(f"debug_crop_raw_{i}.png", 255 - raw_crop)

            crop = cv2.bitwise_not(crop_mask)
            if crop.size == 0:
                continue
            inp = self._prep_digit_crop(crop)
            cv2.imwrite(
                f"debug_crop_{i}.png", (inp[0, :, :, 0] * 255).astype(np.uint8)
            )

            try:
                pred = self.digit_model.predict(inp, verbose=0)
                conf = float(np.max(pred))
                label = int(np.argmax(pred))
            except Exception as exc:
                print("[WARN] digit prediction failed:", exc)
                continue

            cv2.rectangle(annotated, (x0b, y0b), (x1b, y1b), (0, 255, 0), 1)
            cv2.putText(
                annotated,
                str(label) if conf >= CONF_THRESH else "?",
                (x0b, y0b - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imwrite("tab_boxes.png", annotated)
        print("[DEBUG] Digit boxes saved → tab_boxes.png")
        # Future-work: stitch multi-digit boxes into full fret numbers

    # -------------------------- NEW: Seek bar helpers --------------------------
    def _format_time(self, scale, value):
        """Return *value* (seconds) as mm:ss for the scale draw."""
        minutes = int(value) // 60
        seconds = int(value) % 60
        return f"{minutes}:{seconds:02d}"

    def _on_seek_button_press(self, widget, event):  # noqa: D401
        """Remember that the user started dragging the seek bar."""
        self.user_is_seeking = True
        return False  # propagate

    def _on_seek_button_release(self, widget, event):  # noqa: D401
        """Perform a rate-preserving seek to the selected position."""
        if not self.pipeline:
            self.user_is_seeking = False
            return False

        target_seconds = self.seek_scale.get_value()
        rate = self.speed_scale.get_value()
        flags = Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE | Gst.SeekFlags.KEY_UNIT
        self.pipeline.seek(
            rate,
            Gst.Format.TIME,
            flags,
            Gst.SeekType.SET,
            int(target_seconds * Gst.SECOND),
            Gst.SeekType.NONE,
            -1,
        )
        self.last_position = int(target_seconds * Gst.SECOND)
        self.user_is_seeking = False
        # Provide visual feedback
        self.update_status(f"Seeking to {self._format_time(None, target_seconds)}")
        return False

    def _update_seek_bar(self):
        """Periodic timer to keep the seek scale in sync with playback."""
        if not self.playbin or self.user_is_seeking:
            return True  # continue timer

        success_pos, position_ns = self.pipeline.query_position(Gst.Format.TIME)
        success_dur, duration_ns = self.pipeline.query_duration(Gst.Format.TIME)

        if success_dur and duration_ns > 0:
            duration_sec = duration_ns / Gst.SECOND
            # Enable and configure scale once duration is known
            if not self.seek_scale.get_sensitive():
                self.seek_scale.set_range(0.0, duration_sec)
                self.seek_scale.set_sensitive(True)
        else:
            duration_sec = None

        if success_pos and duration_sec:
            self.seek_scale.set_value(position_ns / Gst.SECOND)
        return True
    # ---------------------------------------------------------------------------


if __name__ == '__main__':
    print("Starting main...")
    win = GuitarTrainerApp()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    print("Running application...")
    Gtk.main()
    print("Application finished")
