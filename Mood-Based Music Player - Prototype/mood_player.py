"""
Mood-Based Music Player (prototype)

Features:
- Detect mood from webcam using FER (facial expression recognition)
- Detect mood from text using TextBlob sentiment
- Play local music mapped to moods (using pygame mixer)
- Optional: control Spotify playback via Spotipy (requires Spotify credentials + Premium)
- Simple CLI menu

Usage:
1) Install dependencies: pip install -r requirements.txt
2) Prepare local music folders (example structure shown below)
3) (Optional) Create a .env file with SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI
4) Run: python mood_player.py
"""
import os
import random
import time
from pathlib import Path
import sys

try:
    import cv2
    from fer import FER
    from textblob import TextBlob
    import pygame
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    from dotenv import load_dotenv
except Exception as e:
    print("Missing dependencies or import error:", e)
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

# Load environment variables (for Spotify)
load_dotenv()

# ------------------------
# Configuration
# ------------------------
# Map canonical moods to local folders (update these paths to where you keep mp3s)
MOOD_FOLDERS = {
    "happy": "music/happy",
    "sad": "music/sad",
    "angry": "music/angry",
    "surprise": "music/surprise",
    "neutral": "music/neutral",
    "fear": "music/fear",
    "disgust": "music/disgust"
}

# For Spotify
SPOTIFY_SCOPE = "user-modify-playback-state user-read-playback-state user-read-private"


# ------------------------
# Utilities
# ------------------------
def ensure_music_folders():
    """Ensure mood folders exist; create placeholders if not."""
    for mood, path in MOOD_FOLDERS.items():
        p = Path(path)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            # Create placeholder txt to remind user
            (p / "README.txt").write_text(
                f"Place mp3 files for mood '{mood}' in this folder."
            )


def list_music_files(folder):
    p = Path(folder)
    if not p.exists():
        return []
    return [str(x) for x in p.iterdir() if x.suffix.lower() in {".mp3", ".wav", ".ogg", ".flac"}]


# ------------------------
# Mood Detection - Webcam (FER)
# ------------------------
class WebcamMoodDetector:
    def __init__(self, camera_index=0, display=False):
        self.camera_index = camera_index
        self.display = display
        self.detector = FER(mtcnn=True)  # MTCNN face detection often improves results

    def detect_mood(self, num_frames=10):
        """Capture a few frames and return the most frequent emotion."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("Cannot open webcam. Check camera index or permissions.")
            return None

        emotion_votes = []
        frames_taken = 0
        try:
            while frames_taken < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                # FER expects RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.detector.detect_emotions(rgb)
                if results:
                    # choose the first face
                    face_emotions = results[0]["emotions"]
                    top_emotion = max(face_emotions, key=face_emotions.get)
                    emotion_votes.append(top_emotion)
                    if self.display:
                        cv2.putText(frame, f"{top_emotion}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("Mood Detector (press q to stop)", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                frames_taken += 1
                time.sleep(0.1)
        finally:
            cap.release()
            if self.display:
                cv2.destroyAllWindows()

        if not emotion_votes:
            return None

        # pick the most common emotion
        mood = max(set(emotion_votes), key=emotion_votes.count)
        return mood


# ------------------------
# Mood Detection - Text
# ------------------------
class TextMoodDetector:
    def detect_mood(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to +1
        # Map polarity to coarse mood
        if polarity > 0.3:
            return "happy"
        elif polarity < -0.3:
            return "sad"
        else:
            return "neutral"


# ------------------------
# Music Player - Local
# ------------------------
class LocalMusicPlayer:
    def __init__(self):
        pygame.mixer.init()

    def play_file(self, filepath):
        print(f"Playing: {filepath}")
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.5)
        except Exception as e:
            print("Failed to play file:", e)

    def play_random_for_mood(self, mood):
        folder = MOOD_FOLDERS.get(mood)
        if not folder:
            print("No folder mapped for mood:", mood)
            return
        files = list_music_files(folder)
        if not files:
            print(f"No music files found in {folder}. Please add mp3/ogg/wav files there.")
            return
        choice = random.choice(files)
        self.play_file(choice)


# ------------------------
# Spotify Controller (Optional)
# ------------------------
class SpotifyController:
    def __init__(self):
        client_id = os.getenv("SPOTIPY_CLIENT_ID")
        client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
        redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")
        if not (client_id and client_secret and redirect_uri):
            raise RuntimeError(
                "Spotify credentials not found in environment. "
                "Add SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI to .env or env."
            )
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=SPOTIFY_SCOPE
        ))

    def find_playlist_for_mood(self, mood_query):
        """Search playlists with mood_query and return top playlist uri (best-effort)."""
        q = f"{mood_query} playlist"
        results = self.sp.search(q=q, type="playlist", limit=5)
        items = results.get("playlists", {}).get("items", [])
        if not items:
            return None
        # pick the top item
        return items[0]["uri"], items[0]["name"]

    def start_playback_by_uri(self, playlist_uri):
        devices = self.sp.devices()
        device_list = devices.get("devices", [])
        if not device_list:
            raise RuntimeError("No active Spotify devices found. Start Spotify on a device and try again.")
        device_id = device_list[0]["id"]
        self.sp.start_playback(device_id=device_id, context_uri=playlist_uri)
        print("Playback started on device:", device_list[0]["name"])


# ------------------------
# CLI Flow
# ------------------------
def main():
    print("="*40)
    print("Mood-Based Music Player — Prototype")
    print("="*40)
    ensure_music_folders()

    webcam_detector = WebcamMoodDetector(display=False)
    text_detector = TextMoodDetector()
    local_player = LocalMusicPlayer()
    spotify_controller = None
    # check spotify env
    spotify_available = all(os.getenv(k) for k in ("SPOTIPY_CLIENT_ID", "SPOTIPY_CLIENT_SECRET", "SPOTIPY_REDIRECT_URI"))
    if spotify_available:
        try:
            spotify_controller = SpotifyController()
            print("Spotify integration enabled.")
        except Exception as e:
            print("Spotify initialization failed:", e)
            spotify_controller = None

    while True:
        print("\nChoose input method:")
        print("1) Webcam emotion detection")
        print("2) Type your mood/journal (text sentiment)")
        print("3) Manual mood selection")
        print("4) Exit")
        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            print("Detecting mood from webcam... (look at camera)")
            mood = webcam_detector.detect_mood(num_frames=12)
            if not mood:
                print("Could not detect mood from webcam.")
                continue
            print("Detected mood:", mood)

        elif choice == "2":
            text = input("Write how you feel (a few sentences):\n")
            mood = text_detector.detect_mood(text)
            print("Inferred mood from text:", mood)

        elif choice == "3":
            print("Available moods:", ", ".join(MOOD_FOLDERS.keys()))
            mood = input("Type mood: ").strip().lower()
            if mood not in MOOD_FOLDERS:
                print("Mood not recognized. Try again.")
                continue

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid choice.")
            continue

        # Decide playback route
        print("\nPlayback options:")
        print("1) Play local music (default)")
        if spotify_controller:
            print("2) Try Spotify playlists")
        play_choice = input("Choose (1 or 2): ").strip() or "1"

        if play_choice == "2" and spotify_controller:
            try:
                uri_name = spotify_controller.find_playlist_for_mood(mood)
                if not uri_name:
                    print("No Spotify playlist found for mood. Falling back to local playback.")
                    local_player.play_random_for_mood(mood)
                else:
                    uri, name = uri_name
                    print(f"Found playlist: {name} — starting playback...")
                    spotify_controller.start_playback_by_uri(uri)
                    print("Tip: Control playback in your Spotify app.")
            except Exception as e:
                print("Spotify playback error:", e)
                print("Falling back to local music.")
                local_player.play_random_for_mood(mood)
        else:
            local_player.play_random_for_mood(mood)


if __name__ == "__main__":
    main()
