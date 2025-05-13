import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed percent (can go over 100)
engine.setProperty('volume', 1) 

# Change the voice
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

print('-\n' * 5)

def speak(text):
    engine.say(text=text)
    engine.runAndWait()
    engine.stop()

speak('Hello, Welcome to the world of Artificial Intelligence. How can I help you today?')