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
            
            # Create audio elements
            # Use a scaletempo filter to preserve pitch when playback rate changes
            self.audio_filter = Gst.ElementFactory.make("scaletempo", "audiofilter")

            # Optional audio sink (auto select)
            self.audiosink = Gst.ElementFactory.make("autoaudiosink", "audiosink")
            
            if not self.playbin or not self.videosink or not self.audiosink:
                print("Failed to create elements")
                self.update_status("Failed to create video player")
                return
            
            # Get the widget from the sink and add it to our video area
            sink_widget = self.videosink.get_property("widget")
            if sink_widget:
                # Remove any existing children
                for child in self.video_area.get_children():
                    self.video_area.remove(child)
                self.video_area.pack_start(sink_widget, True, True, 0)
                sink_widget.show()
            
            # Set up the sinks
            self.playbin.set_property("video-sink", self.videosink)
            self.playbin.set_property("audio-sink", self.audiosink)
            
            # If scaletempo filter is available, let playbin use it so tempo changes keep pitch
            if self.playbin and self.audio_filter:
                self.playbin.set_property("audio-filter", self.audio_filter)
            
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
            pad.link(self.audiosink.get_static_pad("sink"))

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
            
            if ret[0] == Gst.StateChangeReturn.SUCCESS:
                self.update_status("Video ready to play")
            else:
                self.update_status("Failed to prepare video")
                print(f"Failed to prepare video: {ret}")
        self.download_button.set_sensitive(True)

    def on_play_clicked(self, button):
        if self.pipeline:
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
        if self.pipeline:
            ret = self.pipeline.set_state(Gst.State.PAUSED)
            print(f"Pause state change result: {ret}")
            if ret == Gst.StateChangeReturn.SUCCESS:
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

            # We keep the same start position and let the pipeline play until
            # the end (NONE stop type). Gst.CLOCK_TIME_NONE means 'unknown/∞'.
            result = self.playbin.seek(
                speed,               # new playback rate
                Gst.Format.TIME,
                flags,
                Gst.SeekType.SET,    # start type – absolute position
                position,            # start position (current)
                Gst.SeekType.NONE,   # stop type – play until the end
                0                    # stop position (ignored when NONE)
            )

            if result:
                print("Seek (rate change) successful")
                self.update_status(f"Playback speed: {speed:.2f}x")
            else:
                print("Seek (rate change) failed")
                self.update_status("Failed to change speed")

if __name__ == '__main__':
    print("Starting main...")
    win = GuitarTrainerApp()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    print("Running application...")
    Gtk.main()
    print("Application finished")
