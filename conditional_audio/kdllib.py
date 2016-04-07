# License: BSD 3-clause
# Authors: Kyle Kastner

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import linalg, fftpack
from scipy.cluster.vq import vq
from scipy.io import wavfile
import scipy.signal as sg
import tarfile
import os
import glob
import re
import copy
from collections import Counter
import time
import sys
import pickle
import itertools
try:
    import Queue
except ImportError:
    import queue as Queue
import threading
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib


def soundsc(X, copy=True):
    """
    Approximate implementation of soundsc from MATLAB without the audio playing.

    Parameters
    ----------
    X : ndarray
        Signal to be rescaled

    copy : bool, optional (default=True)
        Whether to make a copy of input signal or operate in place.

    Returns
    -------
    X_sc : ndarray
        (-1, 1) scaled version of X as float32, suitable for writing
        with scipy.io.wavfile
    """
    X = np.array(X, copy=copy)
    X = (X - X.min()) / (X.max() - X.min())
    X = .9 * X
    X = 2 * X - 1
    return X.astype('float32')


def download(url, server_fname, local_fname=None, progress_update_percentage=5,
             bypass_certificate_check=False):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    if bypass_certificate_check:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        u = urllib.urlopen(url, context=ctx)
    else:
        u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            print("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        print("Downloading: %s Bytes: %s" % (server_fname, file_size))
        file_size_dl = 0
        block_sz = int(1E7)
        p = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if (file_size_dl * 100. / file_size) > p:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl *
                                               100. / file_size)
                print(status)
                p += progress_update_percentage




class BlizzardThread(threading.Thread):
    """Blizzard Thread"""
    def __init__(self, queue, out_queue, preproc_fn):
        threading.Thread.__init__(self)
        self.queue = queue
        self.out_queue = out_queue
        self.preproc_fn = preproc_fn

    def run(self):
        while True:
            # Grabs image path from queue
            wav_paths, texts = self.queue.get()
            text_group = texts
            wav_group = [wavfile.read(wp)[1] for wp in wav_paths]
            wav_group = [w.astype('float32') / (2 ** 15) for w in wav_group]
            wav_group = [self.preproc_fn(wi) for wi in wav_group]
            self.out_queue.put((wav_group, text_group))
            self.queue.task_done()


class Blizzard_dataset(object):
    def __init__(self, minibatch_size=2,
                 blizzard_path='/home/kkastner/blizzard_data'):
        self.n_fft = 256
        self.n_step = self.n_fft // 4
        self.blizzard_path = blizzard_path
        # extracted text
        self.text_path = os.path.join(self.blizzard_path, 'train', 'segmented',
                                      'prompts.gui')
        with open(self.text_path, 'r') as f:
            tt = f.readlines()
            wav_names = [t.strip() for t in tt[::3]]
            raw_other = tt[2::3]
            raw_text = [t.strip().lower() for t in tt[1::3]]
            all_symbols = set()
            for rt in raw_text:
                all_symbols = set(list(all_symbols) + list(set(rt)))
            self.wav_names = wav_names
            self.text = raw_text
            self.symbols = sorted(list(all_symbols))
        import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
        raise ValueError()
        # These files come from converting the Blizzard mp3 files to wav,
        # then placing in a directory called blizzard_wav
        self.wav_paths = glob.glob(os.path.join(self.blizzard_path,
                                                'blizzard_wav', '*.wav'))
        self.minibatch_size = minibatch_size
        self._lens = np.array([float(len(t)) for t in self.text])

        # Get only the smallest 50% of files for now
        _cut = np.percentile(self._lens, 5)
        _ind = np.where(self._lens <= _cut)[0]

        self.text = [self.text[i] for i in _ind]
        self.wav_names = [self.wav_names[i] for i in _ind]
        assert len(self.text) == len(self.wav_names)
        final_wav_paths = []
        final_text = []
        final_wav_names = []
        for n, (w, t) in enumerate(zip(self.wav_names, self.text)):
            parts = w.split("chp")
            name = parts[0]
            chapter = [pp for pp in parts[1].split("_") if pp != ''][0]
            for p in self.wav_paths:
                if name in p and chapter in p:
                    final_wav_paths.append(p)
                    final_wav_names.append(w)
                    final_text.append(t)
                    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
                    raise ValueError()
                    break

        # resort into shortest -> longest order
        sorted_inds = np.argsort([len(t) for t in final_text])
        st = [final_text[i] for i in sorted_inds]
        swp = [final_wav_paths[i] for i in sorted_inds]
        swn = [final_wav_names[i] for i in sorted_inds]
        self.wav_names = swn
        self.wav_paths = swp
        self.text = st
        assert len(self.wav_names) == len(self.wav_paths)
        assert len(self.wav_paths) == len(self.text)

        self.n_per_epoch = len(self.wav_paths)
        self.n_samples_seen_ = 0

        self.buffer_size = 2
        self.minibatch_size = minibatch_size
        self.input_qsize = 5
        self.min_input_qsize = 2
        if len(self.wav_paths) % self.minibatch_size != 0:
            print("WARNING: Sample size not an even multiple of minibatch size")
            print("Truncating...")
            self.wav_paths = self.wav_paths[:-(
                len(self.wav_paths) % self.minibatch_size)]
            self.text = self.text[:-(
                len(self.text) % self.minibatch_size)]

        assert len(self.wav_paths) % self.minibatch_size == 0
        assert len(self.text) % self.minibatch_size == 0

        self.grouped_wav_paths = zip(*[iter(self.wav_paths)] *
                                      self.minibatch_size)
        self.grouped_text = zip(*[iter(self.text)] *
                                     self.minibatch_size)
        assert len(self.grouped_wav_paths) == len(self.grouped_text)
        self._init_queues()

    def _init_queues(self):
        # Infinite...
        self.grouped_elements = itertools.cycle(zip(self.grouped_wav_paths,
                                                self.grouped_text))
        self.queue = Queue.Queue()
        self.out_queue = Queue.Queue(maxsize=self.buffer_size)

        for i in range(1):
            self.it = BlizzardThread(self.queue, self.out_queue, self._pre)
            self.it.setDaemon(True)
            self.it.start()

        # Populate queue with some paths to image data
        for n, _ in enumerate(range(self.input_qsize)):
            group = self.grouped_elements.next()
            self.queue.put(group)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return self._step()

    def reset(self):
        self.n_samples_seen_ = 0
        self._init_queues()

    def _step(self):
        if self.n_samples_seen_ >= self.n_per_epoch:
            self.reset()
            raise StopIteration("End of epoch")
        wav_group, text_group = self.out_queue.get()
        self.n_samples_seen_ += self.minibatch_size
        if self.queue.qsize() <= self.min_input_qsize:
            for i in range(self.input_qsize):
                group = self.grouped_elements.next()
                self.queue.put(group)
        return wav_group, text_group

    def _pre(self, x):
        n_fft = self.n_fft
        n_step = self.n_step
        X_stft = stft(x, n_fft, step=n_step)
        # Power spectrum
        X_mag = complex_to_abs(X_stft)
        X_mag = np.log10(X_mag + 1E-9)
        # unwrap phase then take delta
        X_phase = complex_to_angle(X_stft)
        X_phase = np.vstack((np.zeros_like(X_phase[0][None]), X_phase))
        # Adding zeros to make network predict what *delta* in phase makes sense
        X_phase_unwrap = np.unwrap(X_phase, axis=0)
        X_phase_delta = X_phase_unwrap[1:] - X_phase_unwrap[:-1]
        X_mag_phase = np.hstack((X_mag, X_phase_delta))
        return X_mag_phase

    def _re(self, x):
        n_fft = self.n_fft
        n_step = self.n_step
        # X_mag_phase = unscale(x)
        X_mag_phase = x
        X_mag = X_mag_phase[:, :n_fft // 2]
        X_mag = 10 ** X_mag
        X_phase_delta = X_mag_phase[:, n_fft // 2:]
        # Append leading 0s for consistency
        X_phase_delta = np.vstack((np.zeros_like(X_phase_delta[0][None]),
                                   X_phase_delta))
        X_phase = np.cumsum(X_phase_delta, axis=0)[:-1]
        X_stft = abs_and_angle_to_complex(X_mag, X_phase)
        X_r = istft(X_stft, n_fft, step=n_step, wsola=False)
        return X_r

    """
    # Can't figure out how to do this yet...
    # Iterators R hard
    X = [_pre(di) for di in d]
    X_len = np.sum([len(Xi) for Xi in X])
    X_sum = np.sum([Xi.sum(axis=0) for Xi in X], axis=0)
    X_mean = X_sum / X_len
    X_var = np.sum([np.sum((Xi - X_mean[None]) ** 2, axis=0)
                    for Xi in X], axis=0) / X_len

    def scale(x):
        # WARNING: OPERATES IN PLACE!!!
        # Can only realistically scale magnitude...
        # Phase cost depends on circularity
        x = np.copy(x)
        _x = x[:, :n_fft // 2]
        _mean = X_mean[None, :n_fft // 2]
        _var = X_var[None, :n_fft // 2]
        x[:, :n_fft // 2] = (_x - _mean) / _var
        return x

    def unscale(x):
        # WARNING: OPERATES IN PLACE!!!
        # Can only realistically scale magnitude...
        # Phase cost depends on circularity
        x = np.copy(x)
        _x = x[:, :n_fft // 2]
        _mean = X_mean[None, :n_fft // 2]
        _var = X_var[None, :n_fft // 2]
        x[:, :n_fft // 2] = _x * _var + _mean
        return x
    """


def fetch_sample_speech_fruit(n_samples=None):
    url = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
    wav_path = "audio.tar.gz"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    tf = tarfile.open(wav_path)
    wav_names = [fname for fname in tf.getnames()
                 if ".wav" in fname.split(os.sep)[-1]]
    speech = []
    print("Loading speech files...")
    for wav_name in wav_names[:n_samples]:
        f = tf.extractfile(wav_name)
        fs, d = wavfile.read(f)
        d = d.astype('float32') / (2 ** 15)
        speech.append(d)
    return fs, speech, wav_names


def fetch_sample_speech_tapestry():
    url = "https://www.dropbox.com/s/qte66a7haqspq2g/tapestry.wav?dl=1"
    wav_path = "tapestry.wav"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    fs, d = wavfile.read(wav_path)
    d = d.astype('float32') / (2 ** 15)
    # file is stereo? - just choose one channel
    return fs, d


def complex_to_real_view(arr_c):
    # Inplace view from complex to r, i as separate columns
    assert arr_c.dtype in [np.complex64, np.complex128]
    shp = arr_c.shape
    dtype = np.float64 if arr_c.dtype == np.complex128 else np.float32
    arr_r = arr_c.ravel().view(dtype=dtype).reshape(shp[0], 2 * shp[1])
    return arr_r


def real_to_complex_view(arr_r):
    # Inplace view from real, image as columns to complex
    assert arr_r.dtype not in [np.complex64, np.complex128]
    shp = arr_r.shape
    dtype = np.complex128 if arr_r.dtype == np.float64 else np.complex64
    arr_c = arr_r.ravel().view(dtype=dtype).reshape(shp[0], shp[1] // 2)
    return arr_c


def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X

    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap

    window_size : int
        Size of windows to take

    window_step : int
        Step size between windows

    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    overlap_sz = window_size - window_step
    new_shape = X.shape[:-1] + ((X.shape[-1] - overlap_sz) // window_step, window_size)
    new_strides = X.strides[:-1] + (window_step * X.strides[-1],) + X.strides[-1:]
    X_strided = as_strided(X, shape=new_shape, strides=new_strides)
    return X_strided


def halfoverlap(X, window_size):
    """
    Create an overlapped version of X using 50% of window_size as overlap.

    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap

    window_size : int
        Size of windows to take

    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    window_step = window_size // 2
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    num_frames = len(X) // window_step - 1
    row_stride = X.itemsize * window_step
    col_stride = X.itemsize
    X_strided = as_strided(X, shape=(num_frames, window_size),
                           strides=(row_stride, col_stride))
    return X_strided


def invert_halfoverlap(X_strided):
    """
    Invert ``halfoverlap`` function to reconstruct X

    Parameters
    ----------
    X_strided : ndarray, shape=(n_windows, window_size)
        X as overlapped windows

    Returns
    -------
    X : ndarray, shape=(n_samples,)
        Reconstructed version of X
    """
    # Hardcoded 50% overlap! Can generalize later...
    n_rows, n_cols = X_strided.shape
    X = np.zeros((((int(n_rows // 2) + 1) * n_cols),)).astype(X_strided.dtype)
    start_index = 0
    end_index = n_cols
    window_step = n_cols // 2
    for row in range(X_strided.shape[0]):
        X[start_index:end_index] += X_strided[row]
        start_index += window_step
        end_index += window_step
    return X


def overlap_add(X_strided, window_step, wsola=False):
    """
    overlap add to reconstruct X

    Parameters
    ----------
    X_strided : ndarray, shape=(n_windows, window_size)
        X as overlapped windows

    window_step : int
       step size for overlap add

    Returns
    -------
    X : ndarray, shape=(n_samples,)
        Reconstructed version of X
    """
    n_rows, window_size = X_strided.shape

    # Start with largest size (no overlap) then truncate after we finish
    # +2 for one window on each side
    X = np.zeros(((n_rows + 2) * window_size,)).astype(X_strided.dtype)
    start_index = 0

    total_windowing_sum = np.zeros((X.shape[0]))
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(window_size) / (
        window_size - 1))
    for i in range(n_rows):
        end_index = start_index + window_size
        if wsola:
            offset_size = window_size - window_step
            offset = xcorr_offset(X[start_index:start_index + offset_size],
                                  X_strided[i, :offset_size])
            ss = start_index - offset
            st = end_index - offset
            if start_index - offset < 0:
                ss = 0
                st = 0 + (end_index - start_index)
            X[ss:st] += X_strided[i]
            total_windowing_sum[ss:st] += win
            start_index = start_index + window_step
        else:
            X[start_index:end_index] += X_strided[i]
            total_windowing_sum[start_index:end_index] += win
            start_index += window_step
    # Not using this right now
    #X = np.real(X) / (total_windowing_sum + 1)
    X = X[:end_index]
    return X


def stft(X, fftsize=128, step="half", mean_normalize=True, real=False,
         compute_onesided=True):
    """
    Compute STFT for 1D real valued input X
    """
    if real:
        local_fft = fftpack.rfft
        cut = -1
    else:
        local_fft = fftpack.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2
    if mean_normalize:
        X -= X.mean()
    if step == "half":
        X = halfoverlap(X, fftsize)
    else:
        X = overlap(X, fftsize, step)
    size = fftsize
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    X = X * win[None]
    X = local_fft(X)[:, :cut]
    return X


def istft(X, fftsize=128, step="half", wsola=False, mean_normalize=True,
          real=False, compute_onesided=True):
    """
    Compute ISTFT for STFT transformed X
    """
    if real:
        local_ifft = fftpack.irfft
        X_pad = np.zeros((X.shape[0], X.shape[1] + 1)) + 0j
        X_pad[:, :-1] = X
        X = X_pad
    else:
        local_ifft = fftpack.ifft
    if compute_onesided:
        X_pad = np.zeros((X.shape[0], 2 * X.shape[1])) + 0j
        X_pad[:, :fftsize // 2] = X
        X_pad[:, fftsize // 2:] = 0
        X = X_pad
    X = local_ifft(X).astype("float64")
    if step == "half":
        X = invert_halfoverlap(X)
    else:
        X = overlap_add(X, step, wsola=wsola)
    if mean_normalize:
        X -= np.mean(X)
    return X


def mdct_slow(X, dctsize=128):
    M = dctsize
    N = 2 * dctsize
    N_0 = (M + 1) / 2
    X = halfoverlap(X, N)
    X = sine_window(X)
    n, k = np.meshgrid(np.arange(N), np.arange(M))
    # Use transpose due to "samples as rows" convention
    tf = np.cos(np.pi * (n + N_0) * (k + 0.5) / M).T
    return np.dot(X, tf)


def imdct_slow(X, dctsize=128):
    M = dctsize
    N = 2 * dctsize
    N_0 = (M + 1) / 2
    N_4 = N / 4
    n, k = np.meshgrid(np.arange(N), np.arange(M))
    # inverse *is not* transposed
    tf = np.cos(np.pi * (n + N_0) * (k + 0.5) / M)
    X_r = np.dot(X, tf) / N_4
    X_r = sine_window(X_r)
    X = invert_halfoverlap(X_r)
    return X


def herz_to_mel(freqs):
    """
    Based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    f_0 = 0  # 133.33333
    f_sp = 200 / 3.  # 66.66667
    bark_freq = 1000.
    bark_pt = (bark_freq - f_0) / f_sp
    # The magic 1.0711703 which is the ratio needed to get from 1000 Hz
    # to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz
    # and the preceding linear filter center at 933.33333 Hz
    # (actually 1000/933.33333 = 1.07142857142857 and
    # exp(log(6.4)/27) = 1.07117028749447)
    if not isinstance(freqs, np.ndarray):
        freqs = np.array(freqs)[None]
    log_step = np.exp(np.log(6.4) / 27)
    lin_pts = (freqs < bark_freq)
    mel = 0. * freqs
    mel[lin_pts] = (freqs[lin_pts] - f_0) / f_sp
    mel[~lin_pts] = bark_pt + np.log(freqs[~lin_pts] / bark_freq) / np.log(
        log_step)
    return mel


def mel_to_herz(mel):
    """
    Based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    f_0 = 0  # 133.33333
    f_sp = 200 / 3.  # 66.66667
    bark_freq = 1000.
    bark_pt = (bark_freq - f_0) / f_sp
    # The magic 1.0711703 which is the ratio needed to get from 1000 Hz
    # to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz
    # and the preceding linear filter center at 933.33333 Hz
    # (actually 1000/933.33333 = 1.07142857142857 and
    # exp(log(6.4)/27) = 1.07117028749447)
    if not isinstance(mel, np.ndarray):
        mel = np.array(mel)[None]
    log_step = np.exp(np.log(6.4) / 27)
    lin_pts = (mel < bark_pt)

    freqs = 0. * mel
    freqs[lin_pts] = f_0 + f_sp * mel[lin_pts]
    freqs[~lin_pts] = bark_freq * np.exp(np.log(log_step) * (
        mel[~lin_pts] - bark_pt))
    return freqs


def mel_freq_weights(n_fft, fs, n_filts=None, width=None):
    """
    Based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    min_freq = 0
    max_freq = fs // 2
    if width is None:
        width = 1.
    if n_filts is None:
        n_filts = int(herz_to_mel(max_freq) / 2) + 1
    else:
        n_filts = int(n_filts)
        assert n_filts > 0
    weights = np.zeros((n_filts, n_fft))
    fft_freqs = np.arange(n_fft // 2) / n_fft * fs
    min_mel = herz_to_mel(min_freq)
    max_mel = herz_to_mel(max_freq)
    partial = np.arange(n_filts + 2) / (n_filts + 1.) * (max_mel - min_mel)
    bin_freqs = mel_to_herz(min_mel + partial)
    bin_bin = np.round(bin_freqs / fs * (n_fft - 1))
    for i in range(n_filts):
        fs_i = bin_freqs[i + np.arange(3)]
        fs_i = fs_i[1] + width * (fs_i - fs_i[1])
        lo_slope = (fft_freqs - fs_i[0]) / float(fs_i[1] - fs_i[0])
        hi_slope = (fs_i[2] - fft_freqs) / float(fs_i[2] - fs_i[1])
        weights[i, :n_fft // 2] = np.maximum(
            0, np.minimum(lo_slope, hi_slope))
    # Constant amplitude multiplier
    weights = np.diag(2. / (bin_freqs[2:n_filts + 2]
                      - bin_freqs[:n_filts])).dot(weights)
    weights[:, n_fft // 2:] = 0
    return weights


def time_attack_agc(X, fs, t_scale=0.5, f_scale=1.):
    """
    AGC based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    # 32 ms grid for FFT
    n_fft = 2 ** int(np.log(0.032 * fs) / np.log(2))
    f_scale = float(f_scale)
    window_size = n_fft
    window_step = window_size // 2
    X_freq = stft(X, window_size, mean_normalize=False)
    fft_fs = fs / window_step
    n_bands = max(10, 20 / f_scale)
    mel_width = f_scale * n_bands / 10.
    f_to_a = mel_freq_weights(n_fft, fs, n_bands, mel_width)
    f_to_a = f_to_a[:, :n_fft // 2]
    audiogram = np.abs(X_freq).dot(f_to_a.T)
    fbg = np.zeros_like(audiogram)
    state = np.zeros((audiogram.shape[1],))
    alpha = np.exp(-(1. / fft_fs) / t_scale)
    for i in range(len(audiogram)):
        state = np.maximum(alpha * state, audiogram[i])
        fbg[i] = state

    sf_to_a = np.sum(f_to_a, axis=0)
    E = np.diag(1. / (sf_to_a + (sf_to_a == 0)))
    E = E.dot(f_to_a.T)
    E = fbg.dot(E.T)
    E[E <= 0] = np.min(E[E > 0])
    ts = istft(X_freq / E, window_size, mean_normalize=False)
    return ts, X_freq, E


def sine_window(X):
    """
    Apply a sinusoid window to X.

    Parameters
    ----------
    X : ndarray, shape=(n_samples, n_features)
        Input array of samples

    Returns
    -------
    X_windowed : ndarray, shape=(n_samples, n_features)
        Windowed version of X.
    """
    i = np.arange(X.shape[1])
    win = np.sin(np.pi * (i + 0.5) / X.shape[1])
    row_stride = 0
    col_stride = win.itemsize
    strided_win = as_strided(win, shape=X.shape,
                             strides=(row_stride, col_stride))
    return X * strided_win


def complex_to_abs(arr_c):
    return np.abs(arr_c)


def complex_to_angle(arr_c):
    return np.angle(arr_c)


def abs_and_angle_to_complex(arr_abs, arr_angle):
    # abs(f_c2 - f_c) < 1E-15
    return arr_abs * np.exp(1j * arr_angle)


def angle_to_sin_cos(arr_angle):
    return np.hstack((np.sin(arr_angle), np.cos(arr_angle)))


def sin_cos_to_angle(arr_sin, arr_cos):
    return np.arctan2(arr_sin, arr_cos)


def xcorr_offset(x1, x2):
    """
    Under MSR-LA License

    Based on MATLAB implementation from Spectrogram Inversion Toolbox

    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.

    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.

    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    frame_size = len(x2)
    half = frame_size // 2
    corrs = np.convolve(x1.astype('float32'), x2[::-1].astype('float32'))
    corrs[:half] = -1E30
    corrs[-half:] = -1E30
    offset = corrs.argmax() - len(x1)
    return offset


def invert_spectrogram(X_s, step, calculate_offset=True, set_zero_phase=True):
    """
    Under MSR-LA License

    Based on MATLAB implementation from Spectrogram Inversion Toolbox

    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.

    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.

    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    size = int(X_s.shape[1] // 2)
    wave = np.zeros((X_s.shape[0] * step + size))
    # Getting overflow warnings with 32 bit...
    wave = wave.astype('float64')
    total_windowing_sum = np.zeros((X_s.shape[0] * step + size))
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))

    est_start = int(size // 2) - 1
    est_end = est_start + size
    for i in range(X_s.shape[0]):
        wave_start = int(step * i)
        wave_end = wave_start + size
        if set_zero_phase:
            spectral_slice = X_s[i].real + 0j
        else:
            # already complex
            spectral_slice = X_s[i]

        # Don't need fftshift due to different impl.
        wave_est = np.real(np.fft.ifft(spectral_slice))[::-1]
        if calculate_offset and i > 0:
            offset_size = size - step
            if offset_size <= 0:
                print("WARNING: Large step size >50\% detected! "
                      "This code works best with high overlap - try "
                      "with 75% or greater")
                offset_size = step
            offset = xcorr_offset(wave[wave_start:wave_start + offset_size],
                                  wave_est[est_start:est_start + offset_size])
        else:
            offset = 0
        wave[wave_start:wave_end] += win * wave_est[
            est_start - offset:est_end - offset]
        total_windowing_sum[wave_start:wave_end] += win
    wave = np.real(wave) / (total_windowing_sum + 1E-6)
    return wave


def iterate_invert_spectrogram(X_s, fftsize, step, n_iter=10, verbose=False,
                               complex_input=False):
    """
    Under MSR-LA License

    Based on MATLAB implementation from Spectrogram Inversion Toolbox

    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.

    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.

    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    reg = np.max(X_s) / 1E8
    X_best = copy.deepcopy(X_s)
    for i in range(n_iter):
        if verbose:
            print("Runnning iter %i" % i)
        if i == 0 and not complex_input:
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=True)
        else:
            # Calculate offset was False in the MATLAB version
            # but in mine it massively improves the result
            # Possible bug in my impl?
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=False)
        est = stft(X_t, fftsize=fftsize, step=step, compute_onesided=False)
        phase = est / np.maximum(reg, np.abs(est))
        X_best = X_s * phase[:len(X_s)]
    X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                             set_zero_phase=False)
    return np.real(X_t)


def voiced_unvoiced(X, window_size=256, window_step=128, copy=True):
    """
    Voiced unvoiced detection from a raw signal

    Based on code from:
        https://www.clear.rice.edu/elec532/PROJECTS96/lpc/code.html

    Other references:
        http://www.seas.ucla.edu/spapl/code/harmfreq_MOLRT_VAD.m

    Parameters
    ----------
    X : ndarray
        Raw input signal

    window_size : int, optional (default=256)
        The window size to use, in samples.

    window_step : int, optional (default=128)
        How far the window steps after each calculation, in samples.

    copy : bool, optional (default=True)
        Whether to make a copy of the input array or allow in place changes.
    """
    X = np.array(X, copy=copy)
    if len(X.shape) < 2:
        X = X[None]
    n_points = X.shape[1]
    n_windows = n_points // window_step
    # Padding
    pad_sizes = [(window_size - window_step) // 2,
                 window_size - window_step // 2]
    # TODO: Handling for odd window sizes / steps
    X = np.hstack((np.zeros((X.shape[0], pad_sizes[0])), X,
                   np.zeros((X.shape[0], pad_sizes[1]))))

    clipping_factor = 0.6
    b, a = sg.butter(10, np.pi * 9 / 40)
    voiced_unvoiced = np.zeros((n_windows, 1))
    period = np.zeros((n_windows, 1))
    for window in range(max(n_windows - 1, 1)):
        XX = X.ravel()[window * window_step + np.arange(window_size)]
        XX *= sg.hamming(len(XX))
        XX = sg.lfilter(b, a, XX)
        left_max = np.max(np.abs(XX[:len(XX) // 3]))
        right_max = np.max(np.abs(XX[-len(XX) // 3:]))
        clip_value = clipping_factor * np.min([left_max, right_max])
        XX_clip = np.clip(XX, clip_value, -clip_value)
        XX_corr = np.correlate(XX_clip, XX_clip, mode='full')
        center = np.argmax(XX_corr)
        right_XX_corr = XX_corr[center:]
        prev_window = max([window - 1, 0])
        if voiced_unvoiced[prev_window] > 0:
            # Want it to be harder to turn off than turn on
            strength_factor = .29
        else:
            strength_factor = .3
        start = np.where(right_XX_corr < .3 * XX_corr[center])[0]
        # 20 is hardcoded but should depend on samplerate?
        try:
            start = np.max([20, start[0]])
        except IndexError:
            start = 20
        search_corr = right_XX_corr[start:]
        index = np.argmax(search_corr)
        second_max = search_corr[index]
        if (second_max > strength_factor * XX_corr[center]):
            voiced_unvoiced[window] = 1
            period[window] = start + index - 1
        else:
            voiced_unvoiced[window] = 0
            period[window] = 0
    return np.array(voiced_unvoiced), np.array(period)


def lpc_analysis(X, order=8, window_step=128, window_size=2 * 128,
                 emphasis=0.9, voiced_start_threshold=.9,
                 voiced_stop_threshold=.6, truncate=False, copy=True):
    """
    Extract LPC coefficients from a signal

    Based on code from:
        http://labrosa.ee.columbia.edu/matlab/sws/

    _rParameters
    ----------
    X : ndarray
        Signals to extract LPC coefficients from

    order : int, optional (default=8)
        Order of the LPC coefficients. For speech, use the general rule that the
        order is two times the expected number of formants plus 2.
        This can be formulated as 2 + 2 * (fs // 2000). For approx. signals
        with fs = 7000, this is 8 coefficients - 2 + 2 * (7000 // 2000).

    window_step : int, optional (default=128)
        The size (in samples) of the space between each window

    window_size : int, optional (default=2 * 128)
        The size of each window (in samples) to extract coefficients over

    emphasis : float, optional (default=0.9)
        The emphasis coefficient to use for filtering

    voiced_start_threshold : float, optional (default=0.9)
        Upper power threshold for estimating when speech has started

    voiced_stop_threshold : float, optional (default=0.6)
        Lower power threshold for estimating when speech has stopped

    truncate : bool, optional (default=False)
        Whether to cut the data at the last window or do zero padding.

    copy : bool, optional (default=True)
        Whether to copy the input X or modify in place

    Returns
    -------
    lp_coefficients : ndarray
        lp coefficients to describe the frame

    per_frame_gain : ndarray
        calculated gain for each frame

    residual_excitation : ndarray
        leftover energy which is not described by lp coefficents and gain

    voiced_frames : ndarray
        array of [0, 1] values which holds voiced/unvoiced decision for each
        frame.

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    X = np.array(X, copy=copy)
    if len(X.shape) < 2:
        X = X[None]

    n_points = X.shape[1]
    n_windows = n_points // window_step
    if not truncate:
        pad_sizes = [(window_size - window_step) // 2,
                     window_size - window_step // 2]
        # TODO: Handling for odd window sizes / steps
        X = np.hstack((np.zeros((X.shape[0], pad_sizes[0])), X,
                       np.zeros((X.shape[0], pad_sizes[1]))))
    else:
        pad_sizes = [0, 0]
        X = X[0, :n_windows * window_step]

    lp_coefficients = np.zeros((n_windows, order + 1))
    per_frame_gain = np.zeros((n_windows, 1))
    residual_excitation = np.zeros(
        ((n_windows - 1) * window_step + window_size))
    # Pre-emphasis high-pass filter
    X = sg.lfilter([1, -emphasis], 1, X)
    # stride_tricks.as_strided?
    autocorr_X = np.zeros((n_windows, 2 * window_size - 1))
    for window in range(max(n_windows - 1, 1)):
        XX = X.ravel()[window * window_step + np.arange(window_size)]
        WXX = XX * sg.hanning(window_size)
        autocorr_X[window] = np.correlate(WXX, WXX, mode='full')
        center = np.argmax(autocorr_X[window])
        RXX = autocorr_X[window,
                         np.arange(center, window_size + order)]
        R = linalg.toeplitz(RXX[:-1])
        solved_R = linalg.pinv(R).dot(RXX[1:])
        filter_coefs = np.hstack((1, -solved_R))
        residual_signal = sg.lfilter(filter_coefs, 1, WXX)
        gain = np.sqrt(np.mean(residual_signal ** 2))
        lp_coefficients[window] = filter_coefs
        per_frame_gain[window] = gain
        assign_range = window * window_step + np.arange(window_size)
        residual_excitation[assign_range] += residual_signal / gain
    # Throw away first part in overlap mode for proper synthesis
    residual_excitation = residual_excitation[pad_sizes[0]:]
    return lp_coefficients, per_frame_gain, residual_excitation


def lpc_synthesis(lp_coefficients, per_frame_gain, residual_excitation=None,
                  voiced_frames=None, window_step=128, emphasis=0.9):
    """
    Synthesize a signal from LPC coefficients

    Based on code from:
        http://labrosa.ee.columbia.edu/matlab/sws/
        http://web.uvic.ca/~tyoon/resource/auditorytoolbox/auditorytoolbox/synlpc.html

    Parameters
    ----------
    lp_coefficients : ndarray
        Linear prediction coefficients

    per_frame_gain : ndarray
        Gain coefficients

    residual_excitation : ndarray or None, optional (default=None)
        Residual excitations. If None, this will be synthesized with white noise

    voiced_frames : ndarray or None, optional (default=None)
        Voiced frames. If None, all frames assumed to be voiced.

    window_step : int, optional (default=128)
        The size (in samples) of the space between each window

    emphasis : float, optional (default=0.9)
        The emphasis coefficient to use for filtering

    overlap_add : bool, optional (default=True)
        What type of processing to use when joining windows

    copy : bool, optional (default=True)
       Whether to copy the input X or modify in place

    Returns
    -------
    synthesized : ndarray
        Sound vector synthesized from input arguments

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    # TODO: Incorporate better synthesis from
    # http://eecs.oregonstate.edu/education/docs/ece352/CompleteManual.pdf
    window_size = 2 * window_step
    [n_windows, order] = lp_coefficients.shape

    n_points = (n_windows + 1) * window_step
    n_excitation_points = n_points + window_step + window_step // 2

    random_state = np.random.RandomState(1999)
    if residual_excitation is None:
        # Need to generate excitation
        if voiced_frames is None:
            # No voiced/unvoiced info, so just use randn
            voiced_frames = np.ones((lp_coefficients.shape[0], 1))
        residual_excitation = np.zeros((n_excitation_points))
        f, m = lpc_to_frequency(lp_coefficients, per_frame_gain)
        t = np.linspace(0, 1, window_size, endpoint=False)
        hanning = sg.hanning(window_size)
        for window in range(n_windows):
            window_base = window * window_step
            index = window_base + np.arange(window_size)
            if voiced_frames[window]:
                sig = np.zeros_like(t)
                cycles = np.cumsum(f[window][0] * t)
                sig += sg.sawtooth(cycles, 0.001)
                residual_excitation[index] += hanning * sig
            residual_excitation[index] += hanning * 0.01 * random_state.randn(
                window_size)
    else:
        n_excitation_points = residual_excitation.shape[0]
        n_points = n_excitation_points + window_step + window_step // 2
    residual_excitation = np.hstack((residual_excitation,
                                     np.zeros(window_size)))
    if voiced_frames is None:
        voiced_frames = np.ones_like(per_frame_gain)

    synthesized = np.zeros((n_points))
    for window in range(n_windows):
        window_base = window * window_step
        oldbit = synthesized[window_base + np.arange(window_step)]
        w_coefs = lp_coefficients[window]
        if not np.all(w_coefs):
            # Hack to make lfilter avoid
            # ValueError: BUG: filter coefficient a[0] == 0 not supported yet
            # when all coeffs are 0
            w_coefs = [1]
        g_coefs = voiced_frames[window] * per_frame_gain[window]
        index = window_base + np.arange(window_size)
        newbit = g_coefs * sg.lfilter([1], w_coefs,
                                      residual_excitation[index])
        synthesized[index] = np.hstack((oldbit, np.zeros(
            (window_size - window_step))))
        synthesized[index] += sg.hanning(window_size) * newbit
    synthesized = sg.lfilter([1], [1, -emphasis], synthesized)
    return synthesized


def run_lpc_example():
    fs, X = fetch_sample_speech_tapestry()
    window_size = 256
    window_step = 128
    a, g, e = lpc_analysis(X, order=8, window_step=window_step,
                           window_size=window_size, emphasis=0.9,
                           copy=True)
    v, p = voiced_unvoiced(X, window_size=window_size,
                           window_step=window_step)
    X_r = lpc_synthesis(a, g, e, voiced_frames=v,
                        emphasis=0.9, window_step=window_step)
    wavfile.write("lpc_orig.wav", fs, soundsc(X))
    wavfile.write("lpc_rec.wav", fs, soundsc(X_r))


def run_fft_dct_example():
    # This is an example of the preproc we want to do with a lot of added noise
    random_state = np.random.RandomState(1999)

    fs, d, _ = fetch_sample_speech_fruit()
    n_fft = 128
    X = d[0]
    X_stft = stft(X, n_fft)
    X_rr = complex_to_real_view(X_stft).ravel()
    X_dct = mdct_slow(X_rr, n_fft)
    """
    X_dct_sub = X_dct[1:] - X_dct[:-1]
    std = X_dct_sub.std(axis=0, keepdims=True)
    X_dct_sub += .15 * std * random_state.randn(
        X_dct_sub.shape[0], X_dct_sub.shape[1])
    X_dct_unsub = np.cumsum(X_dct_sub, axis=0)
    X_idct = imdct_slow(X_dct_unsub, n_fft).reshape(-1, n_fft)
    """
    #std = X_dct.std(axis=0, keepdims=True)
    #X_dct[:, 80:] = 0.
    #X_dct += .8 * std * random_state.randn(
    #    X_dct.shape[0], X_dct.shape[1])
    X_idct = imdct_slow(X_dct, n_fft).reshape(-1, n_fft)
    X_irr = real_to_complex_view(X_idct)
    X_r = istft(X_irr, n_fft)[:len(X)]
    X_r = X_r - X_r.mean()

    SNR = 20 * np.log10(np.linalg.norm(X - X_r) / np.linalg.norm(X))
    print(SNR)

    wavfile.write("fftdct_orig.wav", fs, soundsc(X))
    wavfile.write("fftdct_rec.wav", fs, soundsc(X_r))


class base_iterator(object):
    def __init__(self, list_of_containers, minibatch_size,
                 axis,
                 start_index=0,
                 stop_index=np.inf,
                 randomize=False,
                 make_mask=False,
                 one_hot_class_size=None):
        self.list_of_containers = list_of_containers
        self.minibatch_size = minibatch_size
        self.make_mask = make_mask
        self.start_index = start_index
        self.stop_index = stop_index
        self.randomize = randomize
        self.slice_start_ = start_index
        self.axis = axis
        if axis not in [0, 1]:
            raise ValueError("Unknown sample_axis setting %i" % axis)
        self.one_hot_class_size = one_hot_class_size
        len0 = len(list_of_containers[0])
        assert all([len(ci) == len0 for ci in list_of_containers])
        if one_hot_class_size is not None:
            assert len(self.one_hot_class_size) == len(list_of_containers)

    def reset(self):
        self.slice_start_ = self.start_index
        if self.randomize:
            stop_ind = min(len(self.list_of_containers[0]), self.stop_index)
            start_ind = max(0, self.start_index)
            inds = np.arange(start_ind, stop_ind)
            new_list_of_containers = []
            for ci in self.list_of_containers:
                nci = [ci[i] for i in inds]
                if isinstance(ci, np.ndarray):
                    nci = np.array(nci)
                new_list_of_containers.append(nci)
            self.list_of_containers = new_list_of_containers

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        self.slice_end_ = self.slice_start_ + self.minibatch_size
        if self.slice_end_ > self.stop_index:
            # TODO: Think about boundary issues with weird shaped last mb
            self.reset()
            raise StopIteration("Stop index reached")
        ind = slice(self.slice_start_, self.slice_end_)
        self.slice_start_ = self.slice_end_
        if self.make_mask is False:
            res = self._slice_without_masks(ind)
            if not all([self.minibatch_size in r.shape for r in res]):
                # TODO: Check that things are even
                self.reset()
                raise StopIteration("Partial slice returned, end of iteration")
            return res
        else:
            res = self._slice_with_masks(ind)
            # TODO: Check that things are even
            if not all([self.minibatch_size in r.shape for r in res]):
                self.reset()
                raise StopIteration("Partial slice returned, end of iteration")
            return res

    def _slice_without_masks(self, ind):
        raise AttributeError("Subclass base_iterator and override this method")

    def _slice_with_masks(self, ind):
        raise AttributeError("Subclass base_iterator and override this method")


class list_iterator(base_iterator):
    def _slice_without_masks(self, ind):
        sliced_c = [np.asarray(c[ind]) for c in self.list_of_containers]
        if min([len(i) for i in sliced_c]) < self.minibatch_size:
            self.reset()
            raise StopIteration("Invalid length slice")
        for n in range(len(sliced_c)):
            sc = sliced_c[n]
            if self.one_hot_class_size is not None:
                convert_it = self.one_hot_class_size[n]
                if convert_it is not None:
                    raise ValueError("One hot conversion not implemented")
            if not isinstance(sc, np.ndarray) or sc.dtype == np.object:
                maxlen = max([len(i) for i in sc])
                # Assume they at least have the same internal dtype
                if len(sc[0].shape) > 1:
                    total_shape = (maxlen, sc[0].shape[1])
                elif len(sc[0].shape) == 1:
                    total_shape = (maxlen, 1)
                else:
                    raise ValueError("Unhandled array size in list")
                if self.axis == 0:
                    raise ValueError("Unsupported axis of iteration")
                    new_sc = np.zeros((len(sc), total_shape[0],
                                       total_shape[1]))
                    new_sc = new_sc.squeeze().astype(sc[0].dtype)
                else:
                    new_sc = np.zeros((total_shape[0], len(sc),
                                       total_shape[1]))
                    new_sc = new_sc.astype(sc[0].dtype)
                    for m, sc_i in enumerate(sc):
                        new_sc[:len(sc_i), m, :] = sc_i
                sliced_c[n] = new_sc
        return sliced_c

    def _slice_with_masks(self, ind):
        cs = self._slice_without_masks(ind)
        if self.axis == 0:
            ms = [np.ones_like(c[:, 0]) for c in cs]
        elif self.axis == 1:
            ms = [np.ones_like(c[:, :, 0]) for c in cs]
        assert len(cs) == len(ms)
        return [i for sublist in list(zip(cs, ms)) for i in sublist]


def get_dataset_dir(dataset_name):
    """ Get dataset directory path """
    return os.sep.join(os.path.realpath(__file__).split
                       (os.sep)[:-1] + [dataset_name])



def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    labels_shape = labels_dense.shape
    labels_dense = labels_dense.reshape([-1])
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    labels_one_hot = labels_one_hot.reshape(labels_shape+(num_classes,))
    return labels_one_hot


def tokenize_ind(phrase, vocabulary):
    vocabulary_size = len(vocabulary.keys())
    phrase = [vocabulary[char_] for char_ in phrase]
    phrase = np.array(phrase, dtype='int32').ravel()
    phrase = dense_to_one_hot(phrase, vocabulary_size)
    return phrase


def fetch_fruitspeech():
    fs, d, wav_names = fetch_sample_speech_fruit()
    def matcher(name):
        return name.split("/")[1]

    classes = [matcher(wav_name) for wav_name in wav_names]
    all_chars = [c for c in sorted(list(set("".join(classes))))]
    char2code = {v: k for k, v in enumerate(all_chars)}
    vocabulary_size = len(char2code.keys())
    y = []
    for n, cl in enumerate(classes):
        y.append(tokenize_ind(cl, char2code))

    n_fft = 128
    n_step = n_fft // 4

    def _pre(x):
        X_stft = stft(x, n_fft, step=n_step)
        # Power spectrum
        X_mag = complex_to_abs(X_stft)
        X_mag = np.log10(X_mag + 1E-9)
        # unwrap phase then take delta
        X_phase = complex_to_angle(X_stft)
        X_phase = np.vstack((np.zeros_like(X_phase[0][None]), X_phase))
        # Adding zeros to make network predict what *delta* in phase makes sense
        X_phase_unwrap = np.unwrap(X_phase, axis=0)
        X_phase_delta = X_phase_unwrap[1:] - X_phase_unwrap[:-1]
        X_mag_phase = np.hstack((X_mag, X_phase_delta))
        return X_mag_phase

    X = [_pre(di) for di in d]

    X_len = np.sum([len(Xi) for Xi in X])
    X_sum = np.sum([Xi.sum(axis=0) for Xi in X], axis=0)
    X_mean = X_sum / X_len
    X_var = np.sum([np.sum((Xi - X_mean[None]) ** 2, axis=0)
                    for Xi in X], axis=0) / X_len

    def scale(x):
        # WARNING: OPERATES IN PLACE!!!
        # Can only realistically scale magnitude...
        # Phase cost depends on circularity
        x = np.copy(x)
        _x = x[:, :n_fft // 2]
        _mean = X_mean[None, :n_fft // 2]
        _var = X_var[None, :n_fft // 2]
        x[:, :n_fft // 2] = (_x - _mean) / _var
        return x

    def unscale(x):
        # WARNING: OPERATES IN PLACE!!!
        # Can only realistically scale magnitude...
        # Phase cost depends on circularity
        x = np.copy(x)
        _x = x[:, :n_fft // 2]
        _mean = X_mean[None, :n_fft // 2]
        _var = X_var[None, :n_fft // 2]
        x[:, :n_fft // 2] = _x * _var + _mean
        return x

    X = [scale(Xi) for Xi in X]

    def _re(x):
        X_mag_phase = unscale(x)
        X_mag = X_mag_phase[:, :n_fft // 2]
        X_mag = 10 ** X_mag
        X_phase_delta = X_mag_phase[:, n_fft // 2:]
        # Append leading 0s for consistency
        X_phase_delta = np.vstack((np.zeros_like(X_phase_delta[0][None]),
                                   X_phase_delta))
        X_phase = np.cumsum(X_phase_delta, axis=0)[:-1]
        X_stft = abs_and_angle_to_complex(X_mag, X_phase)
        X_r = istft(X_stft, n_fft, step=n_step, wsola=False)
        return X_r

    """
    for n, Xi in enumerate(X[::8]):
        di = _re(Xi)
        wavfile.write("t_%i.wav" % n, fs, soundsc(di))

    raise ValueError()
    """

    speech = {}
    speech["vocabulary_size"] = vocabulary_size
    speech["vocabulary"] = char2code
    speech["sample_rate"] = fs
    speech["data"] = X
    speech["target"] = y
    speech["reconstruct"] = _re
    return speech


def fetch_fruitspeech_nonpar():
    fs, d, wav_names = fetch_sample_speech_fruit()
    def matcher(name):
        return name.split("/")[1]

    classes = [matcher(wav_name) for wav_name in wav_names]
    all_chars = [c for c in sorted(list(set("".join(classes))))]
    char2code = {v: k for k, v in enumerate(all_chars)}
    vocabulary_size = len(char2code.keys())
    y = []
    for n, cl in enumerate(classes):
        y.append(tokenize_ind(cl, char2code))

    # 256 @ 8khz - .032
    ws = 2 ** int(np.log(0.032 * fs) / np.log(2))
    window_size = ws
    window_step = int(.15 * window_size)
    lpc_order = 30
    def _pre(x):
        a, g, e = lpc_analysis(x, order=lpc_order, window_step=window_step,
                               window_size=window_size, emphasis=0.9,
                               copy=True)
        f_sub = a[:, 1:]
        f_full = stft(x, window_size, window_step) #, compute_onesided=False)
        """
        v, p = voiced_unvoiced(x, window_size=window_size,
                               window_step=window_step)
        cut_len = e.shape[0] - e.shape[0] % len(a)
        e = e[:cut_len]
        e = e.reshape((len(a), -1))
        f_full = np.hstack((a, g, v, e))
        """
        if len(f_sub) >= len(f_full):
            f_sub = f_sub[:len(f_full)]
        else:
            f_full = f_full[:len(f_sub)]
        return f_sub, f_full

    def _train(list_of_data):
        f_sub = None
        f_full = None
        for i in range(len(list_of_data)):
            f_sub_i, f_full_i = _pre(list_of_data[i])
            if f_sub is None:
                f_sub = f_sub_i
                f_full = f_full_i
            else:
                f_sub = np.vstack((f_sub, f_sub_i))
                if f_full.shape[1] > f_full_i.shape[1]:
                    f_full_i = np.hstack(
                        (f_full_i, np.zeros_like(f_full_i[:, -1][:, None])))
                elif f_full_i.shape[1] > f_full.shape[1]:
                    f_full_i = f_full_i[:, :f_full.shape[1]]
                f_full = np.vstack((f_full, f_full_i))
        sub_clusters = f_sub
        full_clusters = f_full
        return sub_clusters, full_clusters

    def _clust(x, sub_clusters, extras=None):
        f_sub, f_full = _pre(x)
        f_clust = f_sub
        mem, _ = vq(copy.deepcopy(f_clust), copy.deepcopy(sub_clusters))
        # scipy vq sometimes puts out garbage? choose one at random...
        # possibly related to NaN in input
        #mem[np.abs(mem) >= len(mel_clusters)] = mem[
        #    np.abs(mem) >= len(mel_clusters)] % len(mel_clusters)
        return mem

    def _re(x, sub_clusters, full_clusters):
        memberships = x
        vq_x = full_clusters[memberships]
        """
        # STFT frames not working well in rec
        x_r = iterate_invert_spectrogram(vq_x, window_size, window_step,
                                         n_iter=50, complex_input=True)
        """
        x_r = istft(vq_x, window_size, window_step, wsola=True)
        """
        a = vq_x[:, :lpc_order + 1]
        offset = lpc_order + 1
        g = vq_x[:, offset:offset + 1]
        offset = offset + 1
        v = vq_x[:, offset:offset + 1]
        offset = offset + 1
        e = vq_x[:, offset:].ravel()
        x_r = lpc_synthesis(a, g, e, voiced_frames=v,
                            emphasis=0.9, window_step=window_step)
        """
        agc_x_r, _, _ = time_attack_agc(x_r, fs)
        return agc_x_r

    random_state = np.random.RandomState(1999)
    all_ind = list(range(8))
    # Get 5 random subsets
    random_state.shuffle(all_ind)
    ind = all_ind[:6]

    d1 = []
    for i in ind:
        d1 += d[i::8]

    sub_clusters, full_clusters = _train(d1)


    def _re_wrap(x):
        x = x.argmax(axis=-1)
        re_d = _re(x, sub_clusters, full_clusters)
        return re_d

    def _apply(x):
        m = _clust(x, sub_clusters)
        return m

    X = [_apply(Xi) for Xi in d]
    X = [dense_to_one_hot(Xi, len(sub_clusters)) for Xi in X]

    """
    for n, Xi in enumerate(X[all_ind[0]::8]):
        di = _re_wrap(Xi)
        wavfile.write("t_%i.wav" % n, fs, soundsc(di))

    for n, Xi in enumerate(X[all_ind[-1]::8]):
        di = _re_wrap(Xi)
        wavfile.write("to_%i.wav" % n, fs, soundsc(di))

    raise ValueError()
    """

    speech = {}
    speech["vocabulary_size"] = vocabulary_size
    speech["vocabulary"] = char2code
    speech["sample_rate"] = fs
    speech["data"] = X
    speech["target"] = y
    speech["reconstruct"] = _re_wrap
    return speech


def plot_lines_iamondb_example(X, title="", save_name=None):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    x = np.cumsum(X[:, 1])
    y = np.cumsum(X[:, 2])

    size_x = x.max() - x.min()
    size_y = y.max() - y.min()

    f.set_size_inches(5 * size_x / size_y, 5)
    cuts = np.where(X[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=1.5)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title(title)

    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, bbox_inches='tight', pad_inches=0)


def implot(arr, title="", cmap="gray", save_name=None):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    ax.matshow(arr, cmap=cmap)
    plt.axis("off")

    def autoaspect(x_range, y_range):
        """
        The aspect to make a plot square with ax.set_aspect in Matplotlib
        """
        mx = max(x_range, y_range)
        mn = min(x_range, y_range)
        if x_range <= y_range:
            return mx / float(mn)
        else:
            return mn / float(mx)

    x1 = arr.shape[0]
    y1 = arr.shape[1]
    asp = autoaspect(x1, y1)
    ax.set_aspect(asp)
    plt.title(title)
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)


def np_zeros(shape):
    return np.zeros(shape).astype(theano.config.floatX)


def np_normal(shape, random_state, scale=0.01):
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape
    return (scale * random_state.randn(*shp)).astype(theano.config.floatX)


def np_ortho(shape, random_state, scale=1.):
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        flat_shp = (shp[0], np.prd(shp[1:]))
    else:
        shp = shape
        flat_shp = shape
    g = random_state.randn(*flat_shp)
    U, S, VT = linalg.svd(g, full_matrices=False)
    res = U if U.shape == flat_shp else VT  # pick one with the correct shape
    res = res.reshape(shp)
    return (scale * res).astype(theano.config.floatX)


def as_shared(arr, name=None):
    """ Quick wrapper for theano.shared """
    if type(arr) in [float, int]:
        if name is not None:
            return theano.shared(np.cast[theano.config.floatX](arr))
        else:
            return theano.shared(np.cast[theano.config.floatX](arr), name=name)
    if name is not None:
        return theano.shared(value=arr, borrow=True)
    else:
        return theano.shared(value=arr, name=name, borrow=True)


def apply_shared(list_of_numpy):
    return [as_shared(arr) for arr in list_of_numpy]


def make_weights(in_dim, out_dims, random_state):
    return apply_shared([np_normal((in_dim, out_dim), random_state)
                         for out_dim in out_dims])


def gru_weights(input_dim, hidden_dim, random_state):
    shape = (input_dim, hidden_dim)
    W = np.hstack([np_normal(shape, random_state),
                   np_normal(shape, random_state),
                   np_normal(shape, random_state)])
    b = np_zeros((3 * shape[1],))
    Wur = np.hstack([np_normal((shape[1], shape[1]), random_state),
                     np_normal((shape[1], shape[1]), random_state), ])
    U = np_normal((shape[1], shape[1]), random_state)
    return W, b, Wur, U


class GRU(object):
    def __init__(self, input_dim, hidden_dim, random_state):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        W, b, Wur, U = gru_weights(input_dim, hidden_dim, random_state)
        self.Wur = as_shared(Wur)
        self.U = as_shared(U)
        self.shape = (input_dim, hidden_dim)

    def get_params(self):
        return self.Wur, self.U

    def step(self, inp, gate_inp, prev_state):
        dim = self.shape[1]
        gates = tensor.nnet.sigmoid(tensor.dot(prev_state, self.Wur) + gate_inp)
        update = gates[:, :dim]
        reset = gates[:, dim:]
        state_reset = prev_state * reset
        next_state = tensor.tanh(tensor.dot(state_reset, self.U) + inp)
        next_state = next_state * update + prev_state * (1 - update)
        return next_state


class GRUFork(object):
    def __init__(self, input_dim, hidden_dim, random_state):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        W, b, Wur, U = gru_weights(input_dim, hidden_dim, random_state)
        self.W = as_shared(W)
        self.b = as_shared(b)
        self.shape = (input_dim, hidden_dim)

    def get_params(self):
        return self.W, self.b

    def proj(self, inp):
        dim = self.shape[1]
        projected = tensor.dot(inp, self.W) + self.b
        if projected.ndim == 3:
            d = projected[:, :, :dim]
            g = projected[:, :, dim:]
        else:
            d = projected[:, :dim]
            g = projected[:, dim:]
        return d, g


def logsumexp(x, axis=None):
    x_max = tensor.max(x, axis=axis, keepdims=True)
    z = tensor.log(tensor.sum(tensor.exp(x - x_max),
                              axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis)


def softmax(X):
    # should work for both 2D and 3D
    dim = X.ndim
    e_X = tensor.exp(X - X.max(axis=dim - 1, keepdims=True))
    out = e_X / e_X.sum(axis=dim - 1, keepdims=True)
    return out


def theano_one_hot(t, r=None):
    if r is None:
        r = tensor.max(t) + 1
    ranges = tensor.shape_padleft(tensor.arange(r), t.ndim)
    return tensor.eq(ranges, tensor.shape_padright(t, 1))


def sample_softmax(coeff, theano_rng, epsilon=1E-5):
    if coeff.ndim > 2:
        raise ValueError("Unsupported dim")
    idx = tensor.argmax(theano_rng.multinomial(pvals=coeff, dtype=coeff.dtype),
                        axis=1)
    return tensor.cast(theano_one_hot(idx, coeff.shape[1]),
                       theano.config.floatX)


def categorical_crossentropy(predicted_values, true_values, eps=0.):
    """
    Multinomial negative log likelihood of predicted compared to one hot
    true_values

    Parameters
    ----------
    predicted_values : tensor, shape 2D or 3D
        The predicted class probabilities out of some layer,
        normally the output of softmax_layer

    true_values : tensor, shape 2D or 3D
        Ground truth one hot values

    eps : float, default 0
        Epsilon to be added during log calculation to avoid NaN values.

    Returns
    -------
    categorical_crossentropy : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D

    """
    indices = tensor.argmax(true_values, axis=-1)
    rows = tensor.arange(true_values.shape[0])
    if eps > 0:
        p = tensor.cast(predicted_values, theano.config.floatX) + eps
        p /= tensor.sum(p, axis=predicted_values.ndim - 1, keepdims=True)
    else:
        p = tensor.cast(predicted_values, theano.config.floatX)
    if predicted_values.ndim < 3:
        return -tensor.log(p)[rows, indices]
    elif predicted_values.ndim == 3:
        d0 = true_values.shape[0]
        d1 = true_values.shape[1]
        pred = p.reshape((d0 * d1, -1))
        ind = indices.reshape((d0 * d1,))
        s = tensor.arange(pred.shape[0])
        correct = -tensor.log(pred)[s, ind]
        return correct.reshape((d0, d1,))
    else:
        raise AttributeError("Tensor dim not supported")


def sample_diagonal_gmm(mu, sigma, coeff, theano_rng, epsilon=1E-5,
                        debug=False):
    if debug:
        idx = tensor.argmax(coeff, axis=1)
    else:
        idx = tensor.argmax(
            theano_rng.multinomial(pvals=coeff, dtype=coeff.dtype), axis=1)
    mu = mu[tensor.arange(mu.shape[0]), :, idx]
    sigma = sigma[tensor.arange(sigma.shape[0]), :, idx]
    if debug:
        z = 0.
    else:
        z = theano_rng.normal(size=mu.shape, avg=0., std=1., dtype=mu.dtype)
    s = mu + sigma * z
    return s


def sample_single_dimensional_gmms(mu, sigma, coeff, theano_rng, epsilon=1E-5,
                                   debug=False):
    if debug:
        idx = tensor.argmax(coeff, axis=1)
    else:
        idx = tensor.argmax(
            theano_rng.multinomial(pvals=coeff, dtype=coeff.dtype), axis=1)
    mu = mu[tensor.arange(mu.shape[0]), :, idx]
    sigma = sigma[tensor.arange(sigma.shape[0]), :, idx]
    if debug:
        z = 0.
    else:
        z = theano_rng.normal(size=mu.shape, avg=0., std=1., dtype=mu.dtype)
    s = mu + sigma * z
    return s


def diagonal_gmm(true, mu, sigma, coeff, epsilon=1E-5):
    n_dim = true.ndim
    shape_t = true.shape
    true = true.reshape((-1, shape_t[-1]))
    true = true.dimshuffle(0, 1, 'x')
    inner = tensor.log(2 * np.pi) + 2 * tensor.log(sigma)
    inner += tensor.sqr((true - mu) / sigma)
    inner = -0.5 * tensor.sum(inner, axis=1)
    nll = -logsumexp(tensor.log(coeff) + inner, axis=1)
    nll = nll.reshape(shape_t[:-1], ndim=n_dim-1)
    return nll


def diagonal_phase_gmm(true, mu, sigma, coeff, epsilon=1E-5):
    n_dim = true.ndim
    shape_t = true.shape
    true = true.reshape((-1, shape_t[-1]))
    true = true.dimshuffle(0, 1, 'x')
    inner0 = np.pi - abs(tensor.mod(true - mu, 2 * np.pi) - np.pi)
    inner = tensor.log(2 * np.pi) + 2 * tensor.log(sigma)
    inner += tensor.sqr(inner0 / sigma)
    inner = -0.5 * tensor.sum(inner, axis=1)
    nll = -logsumexp(tensor.log(coeff) + inner, axis=1)
    nll = nll.reshape(shape_t[:-1], ndim=n_dim-1)
    return nll


def single_dimensional_gmms(true, mu, sigma, coeff, epsilon=1E-5):
    shape_t = true.shape
    true = true.reshape((-1, shape_t[-1]))
    true = true.dimshuffle(0, 1, 'x')
    inner = tensor.log(2 * np.pi) + 2 * tensor.log(sigma)
    inner += tensor.sqr((true - mu) / sigma)
    inner = -0.5 * inner
    nll = -logsumexp(tensor.sum(tensor.log(coeff) + inner, axis=1), axis=1)
    nll = nll.reshape((shape_t[0], shape_t[1]))
    return nll


def single_dimensional_phase_gmms(true, mu, sigma, coeff, epsilon=1E-5):
    shape_t = true.shape
    true = true.reshape((-1, shape_t[-1]))
    true = true.dimshuffle(0, 1, 'x')
    inner0 = np.pi - abs(tensor.mod(true - mu, 2 * np.pi) - np.pi)
    inner = tensor.log(2 * np.pi) + 2 * tensor.log(sigma)
    inner += tensor.sqr(inner0 / sigma)
    inner = -0.5 * inner
    nll = -logsumexp(tensor.sum(tensor.log(coeff) + inner, axis=1), axis=1)
    nll = nll.reshape((shape_t[0], shape_t[1]))
    return nll


def bernoulli_and_bivariate_gmm(true, mu, sigma, corr, coeff, binary,
                                epsilon=1E-5):
    n_dim = true.ndim
    shape_t = true.shape
    true = true.reshape((-1, shape_t[-1]))
    true = true.dimshuffle(0, 1, 'x')

    mu_1 = mu[:, 0, :]
    mu_2 = mu[:, 1, :]

    sigma_1 = sigma[:, 0, :]
    sigma_2 = sigma[:, 1, :]

    binary = (binary + epsilon) * (1 - 2 * epsilon)

    c_b = tensor.sum(tensor.xlogx.xlogy0(true[:, 0],  binary) + tensor.xlogx.xlogy0(
        1 - true[:, 0], 1 - binary), axis=1)

    inner1 = (0.5 * tensor.log(1. - corr ** 2 + epsilon))
    inner1 += tensor.log(sigma_1) + tensor.log(sigma_2)
    inner1 += tensor.log(2. * np.pi)

    t1 = true[:, 1]
    t2 = true[:, 2]
    Z = (((t1 - mu_1)/sigma_1)**2) + (((t2 - mu_2) / sigma_2)**2)
    Z -= (2. * (corr * (t1 - mu_1)*(t2 - mu_2)) / (sigma_1 * sigma_2))
    inner2 = 0.5 * (1. / (1. - corr**2 + epsilon))
    cost = - (inner1 + (inner2 * Z))

    nll = -logsumexp(tensor.log(coeff) + cost, axis=1)
    nll -= c_b
    return nll.reshape(shape_t[:-1], ndim=n_dim-1)


def sample_bernoulli_and_bivariate_gmm(mu, sigma, corr, coeff, binary,
                                       theano_rng, epsilon=1E-5):

    idx = tensor.argmax(theano_rng.multinomial(pvals=coeff, dtype=coeff.dtype),
                        axis=1)

    mu = mu[tensor.arange(mu.shape[0]), :, idx]
    sigma = sigma[tensor.arange(sigma.shape[0]), :, idx]
    corr = corr[tensor.arange(corr.shape[0]), idx]

    mu_x = mu[:, 0]
    mu_y = mu[:, 1]
    sigma_x = sigma[:, 0]
    sigma_y = sigma[:, 1]

    z = theano_rng.normal(size=mu.shape, avg=0., std=1., dtype=mu.dtype)

    un = theano_rng.uniform(size=binary.shape)
    binary = tensor.cast(un < binary, theano.config.floatX)

    s_x = (mu_x + sigma_x * z[:, 0]).dimshuffle(0, 'x')
    s_y = mu_y + sigma_y * (
        (z[:, 0] * corr) + (z[:, 1] * tensor.sqrt(1. - corr ** 2)))
    s_y = s_y.dimshuffle(0, 'x')
    s = tensor.concatenate([binary, s_x, s_y], axis=1)
    return s


def gradient_clipping(grads, rescale=5.):
    grad_norm = tensor.sqrt(sum(map(lambda x: tensor.sqr(x).sum(), grads)))
    scaling_num = rescale
    scaling_den = tensor.maximum(rescale, grad_norm)
    scaling = scaling_num / scaling_den
    return [g * scaling for g in grads]


class adam(object):
    """
    Adam optimizer

    Based on implementation from @NewMu / Alex Radford
    """
    def __init__(self, params, learning_rate, b1=0.1, b2=0.001, eps=1E-8):
        self.learning_rate = as_shared(learning_rate)
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]
        self.velocity_ = [theano.shared(np.zeros_like(p.get_value()))
                          for p in params]
        self.itr_ = theano.shared(np.array(0.).astype(theano.config.floatX))

    def updates(self, params, grads):
        learning_rate = self.learning_rate
        b1 = self.b1
        b2 = self.b2
        eps = self.eps
        updates = []
        itr = self.itr_
        i_t = itr + 1.
        fix1 = 1. - (1. - b1) ** i_t
        fix2 = 1. - (1. - b2) ** i_t
        lr_t = learning_rate * (tensor.sqrt(fix2) / fix1)
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            velocity = self.velocity_[n]
            m_t = (b1 * grad) + ((1. - b1) * memory)
            v_t = (b2 * tensor.sqr(grad)) + ((1. - b2) * velocity)
            g_t = m_t / (tensor.sqrt(v_t) + eps)
            p_t = param - (lr_t * g_t)
            updates.append((memory, m_t))
            updates.append((velocity, v_t))
            updates.append((param, p_t))
        updates.append((itr, i_t))
        return updates


def get_shared_variables_from_function(func):
    shared_variable_indices = [n for n, var in enumerate(func.maker.inputs)
                               if isinstance(var.variable,
                                             theano.compile.SharedVariable)]
    shared_variables = [func.maker.inputs[i].variable
                        for i in shared_variable_indices]
    return shared_variables


def get_values_from_function(func):
    return [v.get_value() for v in get_shared_variables_from_function(func)]


def safe_zip(*args):
    """Like zip, but ensures arguments are of same length.

       Borrowed from pylearn2
    """
    base = len(args[0])
    for i, arg in enumerate(args[1:]):
        if len(arg) != base:
            raise ValueError("Argument 0 has length %d but argument %d has "
                             "length %d" % (base, i+1, len(arg)))
    return zip(*args)


def set_shared_variables_in_function(func, list_of_values):
    # TODO : Add checking that sizes are OK
    shared_variable_indices = [n for n, var in enumerate(func.maker.inputs)
                               if isinstance(var.variable,
                                             theano.compile.SharedVariable)]
    shared_variables = [func.maker.inputs[i].variable
                        for i in shared_variable_indices]
    [s.set_value(v) for s, v in safe_zip(shared_variables, list_of_values)]


def save_weights(save_weights_path, items_dict):
    print("Saving weights to %s" % save_weights_path)
    weights_dict = {}
    # k is the function name, v is a theano function
    for k, v in items_dict.items():
        if isinstance(v, theano.compile.function_module.Function):
            # w is all the numpy values from a function
            w = get_values_from_function(v)
            for n, w_v in enumerate(w):
                weights_dict[k + "_%i" % n] = w_v
    if len(weights_dict.keys()) > 0:
        np.savez(save_weights_path, **weights_dict)
    else:
        print("Possible BUG: no theano functions found in items_dict, "
              "unable to save weights!")


def save_checkpoint(save_path, pickle_item):
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    with open(save_path, mode="wb") as f:
        pickle.dump(pickle_item, f, protocol=-1)
    sys.setrecursionlimit(old_recursion_limit)


def load_checkpoint(saved_checkpoint_path):
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    with open(saved_checkpoint_path, mode="rb") as f:
        pickle_item = pickle.load(f)
    sys.setrecursionlimit(old_recursion_limit)
    return pickle_item

def run_blizzard_example():
    bliz = Blizzard_dataset()
    start = time.time()
    itr = 1
    while True:
        r = bliz.next()
        stop = time.time()
        tot = stop - start
        print("Threaded time: %s" % (tot))
        print("Minibatch %s" % str(itr))
        print("Time ratio (s per minibatch): %s" % (tot / float(itr)))
        itr += 1
        break
    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    raise ValueError()



if __name__ == "__main__":
    #run_fft_dct_example()
    #run_lpc_example()
    run_blizzard_example()
