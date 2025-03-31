import os
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv  # pip install python-dotenv

# Load API key from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI()


def test_chat_completion():
    """Test the chat completion API"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, this is a test!"},
            ],
        )
        print("✅ Chat API test successful:")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Chat API test failed: {e}")
        return False


def test_tts():
    """Test the text-to-speech API"""
    try:
        speech_file_path = Path("test_speech.mp3")

        # Using the recommended streaming response method
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input="This is a test of the OpenAI text-to-speech API.",
        ) as response:
            # Save the audio content to a file
            with open(speech_file_path, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)

        print(f"✅ TTS API test successful: Audio saved to {speech_file_path}")
        return True
    except Exception as e:
        print(f"❌ TTS API test failed: {e}")
        return False


def test_transcription():
    """Test audio transcription (requires an audio file)"""
    # This is optional and will be skipped if no test file exists
    test_file = Path("test_audio.mp3")
    if not test_file.exists():
        print("⚠️ Skipping transcription test: No test_audio.mp3 file found")
        return None

    try:
        with open(test_file, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
        print("✅ Transcription API test successful:")
        print(f"Transcribed text: {transcription.text}")
        return True
    except Exception as e:
        print(f"❌ Transcription API test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing OpenAI API connections...")

    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY") and not client.api_key:
        print("⚠️ No API key found in .env file or environment variables.")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your-api-key-here")
        exit(1)
    else:
        # Only show the first few and last few characters of the API key for verification
        api_key = os.environ.get("OPENAI_API_KEY", client.api_key or "")
        if api_key:
            masked_key = (
                f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
            )
            print(f"Using API key: {masked_key}")

    # Run tests
    chat_result = test_chat_completion()
    tts_result = test_tts()
    transcription_result = test_transcription()

    # Summary
    print("\n--- Test Summary ---")
    print(f"Chat API: {'✅ PASSED' if chat_result else '❌ FAILED'}")
    print(f"TTS API: {'✅ PASSED' if tts_result else '❌ FAILED'}")
    if transcription_result is not None:
        print(
            f"Transcription API: {'✅ PASSED' if transcription_result else '❌ FAILED'}"
        )
    else:
        print("Transcription API: ⚠️ SKIPPED (no test file)")
