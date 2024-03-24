import pyttsx3

engine= pyttsx3.init()

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate',140)


def voice(myText):
    engine.say(myText)
    engine.runAndWait()
