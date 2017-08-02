import numpy
import scipy.fftpack
import librosa
from microphone import record_audio
import math

fs = 44100

# Need 250ms intervals. Since, 44100Hz = 44100 samples/s, (44100/4) cycles/(250ms). Need 44100/4 = 11025 samples.

def file_to_array(file_path):
    """
    It transforms a song into a np array.

    :param
        file_path[String]:
            A file path to the song

    :return:
        samples[np.array]:
            This is an array of the values of the song at the file path at a sampling rate of 44100 Hz.
    """
    samples, fs = librosa.load(file_path, sr=44100, mono=True)
    return samples


def mic_to_numpy_array(time):
    """
    It transforms a mic input into an np array.

    :param
        time[float]:
            The time it needs to record

    :return:
        mic[np.array]:
            This is an array of the values of the song that was recorded at a sampling rate of 44100 Hz.

    """
    mic_input, fs = record_audio(time)
    mic = []
    for i in mic_input:
        mic.append(np.fromstring(i, dtype=np.int16))
    mic = np.hstack(mic)
    return mic

def fft(song_arr):
	"""
	This fast fourier transforms a song array.

	:param
		song_arr[np.array]:
			This is the array that represents the song. 

	:return:
		f[np.array]:
			This is the array of coefficients of the fft. 

	"""

	return np.fft.rfft(song_arr)

def mel_scale(fft_arr):
	"""
	This scales the frequencies from hertz into mels.

	:param
		fft_arr[np.array]:
			This is the array of coefficients of an fft. (Check this).

	:return:
		mels[np.array]:
			This is an array that transformed the fft from Hertz into Mels. 

	"""
	return 2595*np.log10(1 + fft_arr/700)

def dct(mel_arr):
	return scipy.fftpack.dct(mel_arr, n=12)
