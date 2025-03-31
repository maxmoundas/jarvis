# Jarvis

![Jarvis](Jarvis.png)

A voice-controlled AI assistant that uses OpenAI's speech recognition, language model, and text-to-speech capabilities to create a Jarvis-like experience. Inspired by Tony Stark's AI assistant, this voice assistant responds to audio input with a playful British accent and witty personality.

## Features

- **Camera/Remote Button Trigger**: Use your Bluetooth camera remote's volume up button to start/stop recording without touching your device
- **Automatic Recording Limits**: Recordings automatically stop after X number of seconds to conserve API usage
- **Audio Feedback**: Distinct sounds for starting and stopping recordings
- **Conversation History**: Maintains context between interactions
- **Volume Safeguards**: Prevents volume from continuously increasing when using volume button triggers

## Requirements

- Python 3.8+
- OpenAI API key
- macOS (for AppleScript volume control)
- Bluetooth camera remote or other device that controls volume (optional)

## Usage

1. First, test your OpenAI API connection:
   ```
   python test_openai.py
   ```

2. Run the voice assistant:
   ```
   python jarvis_assistant.py
   ```

3. Choose a mode:
   - **Volume Button Trigger Mode**: Uses your Bluetooth remote/camera button to start/stop recording
   - **Manual Mode**: Uses keyboard Enter key to start/stop recording

4. Speak your commands or questions when recording starts
   - Listen for the rising tone indicating recording has started
   - Recording will automatically stop after X number of seconds (listen for falling tone)
   - Or manually stop recording by pressing the volume button/Enter key again

## Customization

You can customize the assistant by modifying these parameters in the `.env` file:

```
# Optional settings
MAX_RECORDING_SECONDS=10
ASSISTANT_VOICE=onyx
ASSISTANT_NAME=Jarvis
```

Available voices include:
- onyx (default, British male voice)
- alloy
- echo
- fable
- nova
- shimmer

## API Usage & Costs

This project uses three OpenAI APIs:
- Speech-to-Text (Whisper API)
- Chat Completion (GPT-4o API)
- Text-to-Speech (TTS API)

Each API call incurs costs. To minimize costs:
- The recording time is limited to 3 seconds by default
- The conversation history is kept in memory to minimize token usage
