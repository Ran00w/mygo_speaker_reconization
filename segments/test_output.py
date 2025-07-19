from email.mime import audio
import os
import pydub
from pydub import AudioSegment

audio = AudioSegment.from_wav("test_part207.wav")
output_file = "test_part207_plus.wav"
audio = audio + 20 * 2  # 20*log10(50) ≈ 34 dB, so add 34 dB
audio.export(output_file, format="wav")