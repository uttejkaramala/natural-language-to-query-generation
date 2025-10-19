import speech_recognition as sr

def voice_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Speak your query...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"📝 You said: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
    except sr.RequestError:
        print("⚠️ Could not request results from Google Speech API.")