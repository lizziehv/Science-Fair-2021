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


# return max freq, min freq, avg freq, number of pauses
def analyze_audio_file(filename):
    # from https://pypi.org/project/crepe/ : "timestamps, predicted fundamental frequency in Hz, voicing confidence,
    # i.e. the confidence in the presence of a pitch"

    audio_sr, audio = wavfile.read(filename)
    audio_time, frequency, confidence, activation = crepe.predict(audio, audio_sr, viterbi=True)
    max_freq, min_freq, avg_freq = max(frequency), min(frequency), sum(frequency)/len(frequency)

    # detect silence in audio
    audio = AudioSegment.from_wav(filename)
    silence_in_audio = silence.detect_silence(audio, min_silence_len=1000, silence_thresh=audio.dBFS-16)
    pauses = len(silence_in_audio)

    return max_freq, min_freq, avg_freq, pauses


def analyze_speech(speech):
    tokens = nltk.word_tokenize(speech)

    # recognize parts of speech
    tags = nltk.pos_tag(tokens)
    tags_pos = {pos: 0 for pos in POS}

    for t in tags:
        if t[1] in tags_pos:
            tags_pos[t[1]] += 1

    tags_pos = {pos: tags_pos[pos]/len(tags) for pos in POS}  # get rates

    # analyze sentiment
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(speech)

    return tags_pos, sentiment_scores


def analyze_speech_from_audio(audio_file):
    r = sr.Recognizer()
    r.pause_threshold = PAUSE_THRESHOLD

    with sr.WavFile(audio_file) as source:  # use "test.wav" as the audio source
        audio = r.record(source)  # extract audio data from the file

    speech = r.recognize_google(audio)
    analyze_speech(speech)
