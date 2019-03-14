import sys
import os
import struct
import pyaudio
from pynput.keyboard import Key, Controller

sys.path.append("C:/Users/Niklas/IdeaProjects/Porcupine/binding/python")

from porcupine import Porcupine

library_path = "C:\\Users\\Niklas\\IdeaProjects\\Porcupine\\lib\\windows\\amd64\\libpv_porcupine.dll"  # Path to Porcupine's C library available under lib/${SYSTEM}/${MACHINE}/
model_file_path = 'C:\\Users\\Niklas\\IdeaProjects\\Porcupine\\lib\\common\\porcupine_params.pv'  # It is available at lib/common/porcupine_params.pv
keyword_file_paths = [
	'C:\\Users\\Niklas\\IdeaProjects\\Porcupine\\Scissors_windows.ppn']
sensitivities = [0.2]
handle = Porcupine(library_path, model_file_path, keyword_file_paths=keyword_file_paths, sensitivities=sensitivities)


def get_next_audio_frame():
	pa = pyaudio.PyAudio()
	return pa.open(
		rate=handle.sample_rate,
		channels=1,
		format=pyaudio.paInt16,
		input=True,
		frames_per_buffer=handle.frame_length,
		input_device_index=1)


def run():
	keyboard = Controller()
	audio_stream = get_next_audio_frame()
	pcm = audio_stream.read(handle.frame_length)
	pcm = struct.unpack_from("h" * handle.frame_length, pcm)
	result = handle.process(pcm)
	if result:
		print("detected")
		keyboard.press(Key.space)
		keyboard.release(Key.space)
