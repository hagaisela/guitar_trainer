import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gtk, Gdk, GLib, Gst, GstVideo
import os
import yt_dlp
import threading
import sys
import math
import traceback
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

print("Starting application...")

class GuitarTrainerApp(Gtk.Window):
    def __init__(self):
        print("Initializing GuitarTrainerApp...")
        super().__init__()
        self.set_title("Guitar Tutorial Player")
        self.set_default_size(800, 600)
        
        # Initialize GStreamer
        print("Initializing GStreamer...")
        Gst.init(None)
        self.pipeline = None
        self.playbin = None
        self.videosink = None
        
        # Create main layout
        self.main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add(self.main_box)
        
        # URL entry
        self.url_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
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
        
        # Speed control
        self.speed_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.speed_label = Gtk.Label(label="Playback Speed:")
        self.speed_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.25, 2.0, 0.25)
        self.speed_scale.set_value(1.0)
        self.speed_scale.connect("value-changed", self.on_speed_changed)
        self.speed_box.pack_start(self.speed_label, False, False, 0)
        self.speed_box.pack_start(self.speed_scale, True, True, 0)
        self.main_box.pack_start(self.speed_box, False, False, 0)
        
        # Playback controls
        self.control_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
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
            
            if self.videosink:
                try:
                    # Make sure video frames are displayed in sync with the pipeline clock
                    self.videosink.set_property("sync", True)
                    print("Set videosink sync property to True")
                except TypeError:
                    print("Could not set sync property on videosink")
                    pass  # some versions may not expose the property
            
            # Build an audio sink bin with scaletempo for pitch-preserving time stretching
            audio_sink_desc = (
                "audioconvert ! audioresample ! scaletempo name=st ! tee name=split "
                "split. ! queue max-size-time=0 max-size-buffers=0 ! audioconvert ! audioresample ! autoaudiosink sync=true "
                "split. ! queue ! audioconvert ! audioresample ! capsfilter caps=audio/x-raw,format=F32LE,channels=1 ! "
                "appsink name=pitchsink emit-signals=true sync=false max-buffers=5 drop=true"
            )
            self.audio_sink_bin = Gst.parse_bin_from_description(audio_sink_desc, True)

            # Retrieve the appsink for pitch detection
            self.pitchsink = self.audio_sink_bin.get_by_name("pitchsink")
            if self.pitchsink:
                self.pitchsink.connect("new-sample", self.on_pitch_sample)
            
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
                self.pitch_label.override_color(Gtk.StateFlags.NORMAL, Gdk.RGBA(1, 1, 0, 1))
                overlay.add_overlay(self.pitch_label)

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
                print(f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}")
                self.update_status(f"Player {new_state.value_nick}")

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

            print(f"Changing playback speed to {speed}. Current position: {position}")

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
            GLib.idle_add(self.pitch_label.set_text, f"{note} ({freq_smoothed:.0f} Hz)")
        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def freq_to_note_name(self, freq):
        if freq <= 0 or math.isnan(freq):
            return "-"
        note_num = int(round(12 * math.log2(freq / 440.0) + 69))
        octave = note_num // 12 - 1
        name = self.NOTE_NAMES[note_num % 12]
        return f"{name}{octave}"

if __name__ == '__main__':
    print("Starting main...")
    win = GuitarTrainerApp()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    print("Running application...")
    Gtk.main()
    print("Application finished")
