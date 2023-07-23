import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from model import NN_class
import scipy.signal as signal
import scipy

""" 
    Ariel Radomislensky 211991146
    Nimrod Adar         209388149
"""


def FourierCoeffGen(x):
    N = len(x)
    # make column vector of signal
    x = np.array(x)
    x = np.expand_dims(x, axis=1)
    # make row vector of k
    k = np.arange(N)
    k = np.expand_dims(k, axis=1).T
    # make column vector of n
    n = k.copy().T
    w_0 = 2*np.pi/N
    return 1/N*np.sum(x*np.exp(-1j*k*w_0*n), axis=0)


def FourierSeries(a_k):
    N = len(a_k)
    # make column vector of the coefficients
    a_k = np.array(a_k)
    a_k = np.expand_dims(a_k, axis=1)
    # make column vector of k
    k = np.arange(N)
    k = np.expand_dims(k, axis=1)
    # make row vector of n
    n = k.copy().T
    w_0 = 2*np.pi/N
    return np.sum(a_k*np.exp(1j*k*w_0*n).T, axis=0)


def xcorr(x, y):
    """
    Perform Cross-Correlation on x and y
    x    : 1st signal
    y    : 2nd signal

    returns
    lags : lags of correlation
    corr : coefficients of correlation
    """
    corr = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    return lags, corr


class Inference:
    def __init__(self, model):
        self.model = model

        self.epsilon = 10 ** -12
        self.alpha = 0.997
        self.window_len = 512

        self.K = self.window_len // 2 + 1

        self.M = np.zeros([self.K * 2, 1])
        self.S = np.ones([self.K * 2, 1])

        self.window = np.hamming(self.window_len)[:, np.newaxis]

    def preprocessing(self, mic_buffer, ref_signal):
        frame = np.zeros([2 * self.K, 1], dtype=complex)
        frame[:: 2, :] = mic_buffer[:self.K, :] / (2 ** 15)
        frame[1::2, :] = ref_signal[:self.K, :] / (2 ** 15)

        frame = np.log10(np.maximum(abs(frame), self.epsilon))

        self.M = self.alpha * self.M + (1 - self.alpha) * frame
        self.S = self.alpha * self.S + (1 - self.alpha) * abs(frame) ** 2

        frame = (frame - self.M) / \
            (np.sqrt(self.S - self.M ** 2) + self.epsilon)

        return frame

    def forward(self, mic, ref):
        idx = 0
        output = np.zeros([len(mic), 1], dtype=complex)

        while idx + self.window_len < len(mic):
            # Get input buffers
            def apply_window(
                x): return x[idx:idx+self.window_len] * self.window.flatten()
            mic_buffer = apply_window(mic)  # TODO: complete code here
            ref_buffer = apply_window(ref)  # TODO: complete code here

            # Transform inputs to the frequency domain
            mic_buffer_coeff = FourierCoeffGen(
                mic_buffer)  # TODO: complete code here
            mic_buffer_coeff = np.expand_dims(mic_buffer_coeff, axis=1)
            ref_buffer_coeff = FourierCoeffGen(
                ref_buffer)  # TODO: complete code here
            ref_buffer_coeff = np.expand_dims(ref_buffer_coeff, axis=1)
            # Pre-processing
            frame = self.preprocessing(
                mic_buffer=mic_buffer_coeff, ref_signal=ref_buffer_coeff)

            # Execute Neural Network - using frame, mic_buffer_coeff as inputs
            output_net_coeff = self.model.forward(
                frame, mic_buffer_coeff)  # TODO: complete code here
            output_net = FourierSeries(output_net_coeff.flatten())
            # Overlap and add
            # TODO: complete code here
            output[idx:idx+self.window_len, 0] += output_net * \
                self.window.flatten()

            # Update index
            idx += self.window_len//4  # TODO: complete code here

        return output


if __name__ == "__main__":
    """
    Note!
    if there is a single code line to complete, it is marked as: 
        model = ...# TODO: complete code here
    if there is a code block to complete, it is marked as:
        # ----------
        # TODO: complete code here
        # ----------
    """

    inference_mode = 'our example'  # 'our example' or 'your record'
    if inference_mode == 'our example':
        # Load data - our example
        fs = 48000
        mic = np.memmap(
            "2tFqz9scnUmF2PX04sNXfg_doubletalk_mic_48kHz_new.pcm", dtype='h', mode='r')
        ref = np.memmap(
            "2tFqz9scnUmF2PX04sNXfg_doubletalk_lpb_48kHz_new.pcm", dtype='h', mode='r')
    elif inference_mode == 'your record':
        # Generate your own data - notice that you need to use your speaker and not your headphone
        fs = 48000
        ref = np.memmap(
            "2tFqz9scnUmF2PX04sNXfg_doubletalk_lpb_48kHz_new.pcm", dtype='h', mode='r')  # 48kHz

        # for first time- record your voice from device microphone while playing reference signal from speaker
        myrecording = sd.playrec(ref, fs, channels=1)
        sd.wait()
        myrecording = np.squeeze(myrecording)
        scipy.io.wavfile.write('my_record.pcm', fs, myrecording.astype(
            np.int16))  # save as pcm file

        # # for next time you can use the pcm file you just created with the following code
        # myrecording = np.memmap("my_record.pcm", dtype='h', mode='r')  # 48kHz

        # Cancel noise from device microphone
        myrecording2 = myrecording[len(myrecording)-len(ref):]
        sd.play(myrecording2, fs)
        sd.wait()

        # Cancel delay between signal reference and your recording voice
        myrecording2_norm = myrecording2/np.max(np.abs(myrecording2))
        ref_norm = ref/np.max(np.abs(ref))
        lags, corr = xcorr(myrecording2_norm, ref_norm)
        delay = lags[np.argmax(corr)]
        myrecording2 = myrecording2[delay:]
        ref = ref[:len(myrecording2)]
        mic = myrecording2

    # Downsample data
    # ----------
    # TODO: complete code here
    # downsample by taking every M'th sample
    def downsample(x, M): return x[::M]
    mic = downsample(mic, fs//16000)
    ref = downsample(ref, fs//16000)
    # ----------
    fs = 16000
    # # Play input data
    # sd.play(mic, fs)
    # sd.wait()
    # sd.play(ref, fs)
    # sd.wait()

    # Load model
    model = NN_class()  # TODO: complete code here

    # Define Inference, and run input through model
    inference = Inference(model)  # TODO: complete code here
    output = inference.forward(mic, ref)  # TODO: complete code here

    # Normalize the output
    output /= np.max(np.abs(output))/0.99  # TODO: complete code here

    # Play and plot the output
    # -------------
    # TODO: complete code here
    #       add sd.wait() after each sd.play()
    print('playing...')
    sd.play(np.real(output), fs)
    sd.wait()
    # -------------

    # Save clean output
    print('done!')
    scipy.io.wavfile.write(f'model_out_{inference_mode}.pcm', fs, output.astype(
        np.int16))  # save as pcm file
