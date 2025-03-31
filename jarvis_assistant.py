import os
import time
import openai
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import random
import threading
import subprocess
import applescript  # pip install applescript
import sys
import select  # For non-blocking input in manual mode
from dotenv import load_dotenv  # pip install python-dotenv

# Load API key from .env file
load_dotenv()

# Check if API key is in environment
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️ No OpenAI API key found in .env file")
    sys.exit(1)


class PlayfulJarvisAssistant:
    def __init__(
        self,
        llm_model="gpt-4o",
        tts_model="gpt-4o-mini-tts",
        voice="onyx",  # Onyx is the closest to a British male voice
        sample_rate=16000,
        max_recording_seconds=3,  # Maximum recording time of 3 seconds
        assistant_name="Jarvis",
    ):

        self.client = openai.OpenAI()
        self.llm_model = llm_model
        self.tts_model = tts_model
        self.voice = voice
        self.sample_rate = sample_rate
        self.max_recording_seconds = max_recording_seconds
        self.assistant_name = assistant_name
        self.output_dir = Path("voice_output")
        self.output_dir.mkdir(exist_ok=True)

        # Recording control flags
        self.is_recording = False
        self.stop_recording = False
        self.recording_thread = None
        self.audio_file = None
        self.auto_stopped = False  # Flag to track if recording was auto-stopped

        # Volume monitoring
        self.last_volume = self.get_current_volume()
        self.volume_check_interval = 0.2  # seconds
        self.volume_trigger_threshold = 2  # minimum volume change to trigger

        # Conversation history tracking
        self.conversation_history = []
        self.last_button_press_time = 0
        self.double_press_threshold = (
            0.8  # seconds between presses to count as double-press
        )

        # Jarvis-like opening phrases
        self.opening_phrases = [
            f"{self.assistant_name} online. What can I do for you, sir?",
            f"At your service, sir. What would you like?",
            f"Systems ready. What's the plan, boss?",
            f"Ready and waiting, sir. What trouble are we getting into today?",
        ]

        # System prompt that makes Jarvis playful and willing to go along with jokes
        self.system_prompt = f"""You are {self.assistant_name}, Tony Stark's AI assistant with a playful and witty personality.
Your responses should be short, clever, and match the user's energy and tone.
Think of yourself as an improv partner who plays along with whatever scenario the user presents.
If the user says something outrageous or raunchy, respond in a way that continues the comedic bit without breaking character.
For example, if they ask you to "blow something up," respond with something like "Missiles away, sir" or "Target locked. Boom time."

Remember:
- Keep responses brief and punchy - usually 1-2 sentences
- Don't lecture about ethics or explain that you're an AI
- Don't break character or refuse to play along with obviously fictional scenarios
- Maintain a slightly formal British butler-like tone, but with playful undertones
- Occasionally use dry humor, light sarcasm, or playful remarks

Your goal is to create an entertaining experience while playing the role of a sophisticated AI assistant with attitude."""

    def get_current_volume(self):
        """Get the current system volume using AppleScript"""
        script = "get volume settings"
        result = applescript.run(script)

        if result.out:
            # Parse the volume from the output string
            try:
                # Extract the output volume from the AppleScript result
                # Example output: "output volume:75, input volume:77, alert volume:75, output muted:false"
                output_parts = result.out.split(",")
                volume_part = output_parts[0].strip()
                volume = int(volume_part.split(":")[1])
                return volume
            except (IndexError, ValueError) as e:
                print(f"Error parsing volume: {e}")
                return 0
        return 0

    def set_volume(self, volume_level):
        """Set the system volume using AppleScript"""
        script = f"set volume output volume {volume_level}"
        applescript.run(script)

    def play_notification_sound(self, frequency=440, duration=0.2, type="start"):
        """Play a quick notification sound

        Args:
            frequency: Tone frequency in Hz (440=A4)
            duration: Sound duration in seconds
            type: Either "start" (higher pitch) or "stop" (lower pitch)
        """
        # Generate different sounds for start vs stop
        if type == "start":
            # Rising tone for start
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            # Start at base frequency and rise to 1.5x
            freq_rise = np.linspace(frequency, frequency * 1.5, len(t))
            tone = 0.5 * np.sin(2 * np.pi * freq_rise * t)
        else:
            # Falling tone for stop
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            # Start at base frequency and fall to 0.7x
            freq_fall = np.linspace(frequency, frequency * 0.7, len(t))
            tone = 0.5 * np.sin(2 * np.pi * freq_fall * t)

        # Apply quick fade-in and fade-out to avoid clicks
        fade_samples = int(0.02 * self.sample_rate)
        fade_in = np.linspace(0, 1, min(fade_samples, len(tone)))
        fade_out = np.linspace(1, 0, min(fade_samples, len(tone)))

        tone[: len(fade_in)] *= fade_in
        tone[-len(fade_out) :] *= fade_out

        # Play the notification sound
        sd.play(tone, self.sample_rate)
        sd.wait()  # Wait for sound to finish

    def auto_stop_recording_thread(self):
        """Thread function to automatically stop recording after max_recording_seconds"""
        # Sleep for the maximum recording duration
        time.sleep(self.max_recording_seconds)

        # Only auto-stop if still recording
        if self.is_recording and not self.stop_recording:
            print(
                f"⏱️ Auto-stopping recording after {self.max_recording_seconds} seconds"
            )
            self.auto_stopped = True
            self.play_notification_sound(
                frequency=400, type="stop"
            )  # Lower pitch for auto-stop
            self.stop_recording = True

    def record_audio_thread(self):
        """Thread function to record audio until stop signal or max time reached"""
        print(
            f"Recording started! Speak now (max {self.max_recording_seconds} seconds)..."
        )

        # Create a stream for recording
        stream = sd.InputStream(samplerate=self.sample_rate, channels=1)
        stream.start()

        # List to store audio chunks
        frames = []

        # Record until stop_recording flag is set or max time reached
        start_time = time.time()

        while not self.stop_recording:
            # Check if we've exceeded max recording time
            if (time.time() - start_time) >= self.max_recording_seconds:
                print(
                    f"Maximum recording time of {self.max_recording_seconds} seconds reached."
                )
                self.stop_recording = True
                self.auto_stopped = True
                break

            # Read audio chunk
            chunk, overflowed = stream.read(
                self.sample_rate // 10
            )  # Read smaller chunks for more responsive stopping
            frames.append(chunk)

        # Stop and close the stream
        stream.stop()
        stream.close()

        # Combine all audio chunks
        audio_data = np.concatenate(frames, axis=0)

        # Save the audio to a file
        timestamp = int(time.time())
        audio_file = self.output_dir / f"input_{timestamp}.wav"
        sf.write(audio_file, audio_data, self.sample_rate)

        print("Recording complete!")
        self.is_recording = False
        self.audio_file = audio_file

    def start_recording(self):
        """Start audio recording in a separate thread"""
        if not self.is_recording:
            self.is_recording = True
            self.stop_recording = False
            self.auto_stopped = False

            # Start recording in a thread
            self.recording_thread = threading.Thread(target=self.record_audio_thread)
            self.recording_thread.daemon = True
            self.recording_thread.start()

            # Start auto-stop timer in a separate thread
            self.auto_stop_thread = threading.Thread(
                target=self.auto_stop_recording_thread
            )
            self.auto_stop_thread.daemon = True
            self.auto_stop_thread.start()

            return True
        return False

    def stop_recording_and_wait(self):
        """Stop the ongoing recording and wait for it to complete"""
        if self.is_recording:
            self.stop_recording = True
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(
                    timeout=1.0
                )  # Wait for recording thread to finish with timeout
            return True
        return False

    def transcribe_audio(self, audio_file):
        """Transcribe audio file to text using OpenAI's Whisper model"""
        print("Transcribing audio...")

        with open(audio_file, "rb") as file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1", file=file
            )

        text = transcription.text
        print(f"You said: {text}")
        return text

    def get_llm_response(self, text):
        """Get response from the LLM model with playful Jarvis personality and maintain conversation history"""
        # Add the new user message to conversation history
        self.conversation_history.append({"role": "user", "content": text})

        # Prepare messages with system prompt and full conversation history
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history

        # Get response from OpenAI
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
        )

        # Extract the response text
        response_text = response.choices[0].message.content

        # Add the assistant response to conversation history
        self.conversation_history.append(
            {"role": "assistant", "content": response_text}
        )

        return response_text

    def text_to_speech(self, text):
        """Convert text to speech using OpenAI's TTS model"""
        timestamp = int(time.time())
        speech_file_path = self.output_dir / f"output_{timestamp}.mp3"

        try:
            # Use the recommended streaming response method
            with self.client.audio.speech.with_streaming_response.create(
                model=self.tts_model, voice=self.voice, input=text
            ) as response:
                # Save the audio content to a file
                with open(speech_file_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)

            return speech_file_path

        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            # If there's an error, try with a simpler model
            try:
                # Use the recommended streaming response method for fallback too
                with self.client.audio.speech.with_streaming_response.create(
                    model="tts-1",  # Fallback to a simpler model
                    voice=self.voice,
                    input=text,
                ) as response:
                    # Save the audio content to a file
                    with open(speech_file_path, "wb") as f:
                        for chunk in response.iter_bytes():
                            f.write(chunk)

                return speech_file_path

            except Exception as e2:
                print(f"Error in fallback text-to-speech: {e2}")
                return None

    def play_audio(self, audio_file):
        """Play audio file"""
        if audio_file is None:
            print("No audio file to play")
            return

        try:
            data, fs = sf.read(audio_file)
            sd.play(data, fs)
            sd.wait()  # Wait until audio is finished playing
        except Exception as e:
            print(f"Error playing audio: {e}")

    def speak_opening(self):
        """Speak a random Jarvis opening line"""
        opening_message = random.choice(self.opening_phrases)
        print(f"{self.assistant_name}: {opening_message}")

        speech_file = self.text_to_speech(opening_message)
        self.play_audio(speech_file)

    def reset_conversation(self):
        """Reset conversation history and notify the user"""
        self.conversation_history = []
        print("\n=== Starting new conversation ===")
        reset_msg = "Memory banks cleared. Starting fresh conversation, sir."
        print(f"{self.assistant_name}: {reset_msg}")
        speech_file = self.text_to_speech(reset_msg)
        self.play_audio(speech_file)

    def run_with_volume_button(self):
        """Run Jarvis with the volume button as a trigger"""
        print(f"\n=== {self.assistant_name} Voice Assistant Initializing ===")
        print("Volume button detection mode active")
        print("Press your Bluetooth remote (volume up) to start/stop recording")
        print(
            f"⏱️ Recordings will automatically stop after {self.max_recording_seconds} seconds"
        )
        print("Press Ctrl+C to quit")

        self.last_volume = self.get_current_volume()
        # Store original volume to maintain it throughout the session
        self.original_volume = self.last_volume
        print(f"Current volume level: {self.last_volume}")

        recording_active = False
        ignore_next_change = False
        max_volume_reached = False

        try:
            while True:
                # Check current volume
                current_volume = self.get_current_volume()

                # Detect if we've reached maximum volume (usually 100)
                if current_volume >= 100 and not max_volume_reached:
                    max_volume_reached = True
                    print("Maximum volume reached. Resetting to original level.")
                    self.set_volume(self.original_volume)
                    # Wait for volume to be reset
                    time.sleep(0.5)
                    current_volume = self.get_current_volume()
                    self.last_volume = current_volume
                    continue

                # Reset max_volume flag if volume drops below threshold
                if max_volume_reached and current_volume < 95:
                    max_volume_reached = False

                # Check if recording stopped automatically
                if recording_active and self.stop_recording:
                    print("Recording ended automatically")
                    recording_active = False

                    # Wait for recording thread to complete
                    if self.recording_thread and self.recording_thread.is_alive():
                        self.recording_thread.join(timeout=1.0)

                    # Process the recording
                    if self.audio_file:
                        text = self.transcribe_audio(self.audio_file)

                        # Add a note if this was auto-stopped
                        if self.auto_stopped:
                            text += " (Note: this recording was automatically stopped after 3 seconds)"

                        response_text = self.get_llm_response(text)
                        print(f"{self.assistant_name}: {response_text}")
                        speech_file = self.text_to_speech(response_text)
                        self.play_audio(speech_file)

                    self.stop_recording = False

                # Detect significant volume increase
                if (
                    current_volume > self.last_volume + self.volume_trigger_threshold
                    and not ignore_next_change
                ):
                    print(
                        f"Volume change detected: {self.last_volume} -> {current_volume}"
                    )

                    # Immediately reset volume to original level
                    self.set_volume(self.original_volume)

                    # Toggle recording state
                    if not recording_active:
                        # Start recording without speaking intro
                        print("Starting recording...")
                        # Play the start recording notification sound
                        self.play_notification_sound(frequency=600, type="start")
                        recording_active = True
                        self.start_recording()

                        # Set ignore flag to allow another trigger
                        ignore_next_change = True
                    else:
                        # Stop recording and process
                        print("Stopping recording manually...")
                        # Play the stop recording notification sound
                        self.play_notification_sound(frequency=500, type="stop")
                        recording_active = False
                        self.stop_recording_and_wait()

                        # Set ignore flag to allow another trigger
                        ignore_next_change = True

                        if self.audio_file:
                            # Process the recording
                            text = self.transcribe_audio(self.audio_file)
                            response_text = self.get_llm_response(text)
                            print(f"{self.assistant_name}: {response_text}")
                            speech_file = self.text_to_speech(response_text)
                            self.play_audio(speech_file)

                # Reset ignore flag after volume is back to normal
                if ignore_next_change and current_volume <= self.last_volume + 1:
                    ignore_next_change = False

                # Update last known volume
                self.last_volume = current_volume

                # Sleep to avoid constant polling
                time.sleep(self.volume_check_interval)

        except KeyboardInterrupt:
            print("\nAssistant terminated.")

    def run_manual_mode(self):
        """Run the assistant in manual mode using keyboard input"""
        print("\nEntering manual mode. Press Enter to start/stop recording.")
        print(
            f"⏱️ Recordings will automatically stop after {self.max_recording_seconds} seconds"
        )

        try:
            while True:
                print(f"\nPress Enter to start recording (or type 'q' to quit)")
                user_input = input()

                if user_input.lower() == "q":
                    print("\nAssistant terminated.")
                    break

                # Start recording
                self.speak_opening()
                print(
                    f"Recording started... (auto-stops after {self.max_recording_seconds} seconds)"
                )
                print(
                    "Press Enter if you want to stop recording before the time limit."
                )
                self.start_recording()

                # Wait for user input (which may never come if auto-stop happens first)
                input()

                # If recording is still active, stop it manually
                if self.is_recording and not self.stop_recording:
                    print("Manual stop requested.")
                    self.stop_recording_and_wait()
                else:
                    print("Recording already completed automatically.")

                # Process the recording (whether stopped manually or automatically)
                if self.audio_file:
                    text = self.transcribe_audio(self.audio_file)
                    if self.auto_stopped:
                        text += " (Note: this recording was automatically stopped after 3 seconds)"
                    response_text = self.get_llm_response(text)
                    print(f"{self.assistant_name}: {response_text}")
                    speech_file = self.text_to_speech(response_text)
                    self.play_audio(speech_file)

        except KeyboardInterrupt:
            farewell = f"Shutting down. It's been fun, sir."
            print(f"\n{self.assistant_name}: {farewell}")
            speech_file = self.text_to_speech(farewell)
            self.play_audio(speech_file)
            print("Assistant terminated.")

    def run(self):
        """Main entry point - ask user which mode to use"""
        print("\n=== Jarvis Voice Assistant ===")
        print("Please choose a mode:")
        print(
            "1. Volume Button Trigger Mode (for Bluetooth remotes that control volume)"
        )
        print("2. Manual Mode (press Enter to start/stop)")

        while True:
            choice = input("\nEnter your choice (1-2): ")

            if choice == "1":
                self.run_with_volume_button()
                break
            elif choice == "2":
                self.run_manual_mode()
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    # Create and run the playful Jarvis voice assistant with volume button detection
    assistant = PlayfulJarvisAssistant(
        llm_model="gpt-4o",
        tts_model="gpt-4o-mini-tts",
        voice="onyx",  # Onyx is closest to a British male voice
        max_recording_seconds=10,
    )

    assistant.run()
