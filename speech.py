# andreachain04, danielavidea, dubysmoreno14, catherineromero16, Juancho031
# Purpose: Functions for recording and analyzing audio

# NOTE: this also requires PyAudio because it uses the Microphone
import speech_recognition as sr
import time

import crepe
from scipy.io import wavfile

from pydub import AudioSegment, silence
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# parts of speech of interest for text analysis
POS = [
    'NNS',   # plural nouns
    'PRP',   # personal pronouns
    'PRP$',  # possessive pronouns
    'VB',    # verbs
    'VBD',   # past tense verbs
    'VBG',   # gerund verbs
    'VBN',   # past participle verbs
    'VBP',   # present verbs
    'VBZ',   # 3rd person verbs
    'UH'     # interjection
]

# negative, neutral, positive, compound sentiment
SENTIMENT = ['neg', 'neu', 'pos', 'compound']

AUDIO_FEATURES = ['SPEECH_RATE', 'MAX_FREQ', 'MIN_FREQ', 'AVG_FREQ', 'PAUSES']

PAUSE_THRESHOLD = 0.5


# record response to file name
def get_audio_and_rate(audio_file):
    # obtain audio from the microphone
    r = sr.Recognizer()
    r.pause_threshold = PAUSE_THRESHOLD

    start_time = time.time()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source, timeout=None)
        total_time = (time.time() - start_time) / 60  # in minutes

    # recognize speech using Google Speech Recognition
    try:
        speech = r.recognize_google(audio)
        print("Google Speech Recognition thinks you said: " + "\"" + speech + "\"")
        speech_rate = len(speech.split(" ")) / total_time
        print("Speaking rate (w/m): " + str(speech_rate))

        # save audio file
        with open(audio_file, "wb") as file:
            file.write(audio.get_wav_data())

        return speech, speech_rate
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
