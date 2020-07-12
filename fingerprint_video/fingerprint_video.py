import cv2
import matplotlib.pyplot as plt
import numpy as np

def validate_video(vid, min_length=600, fps=30):
    assert validate_length(vid, min_length)
    assert validate_correctness(vid)
    assert validate_fps(vid, fps)


def validate_length(vid, length):
    """
    Validates the video has the required length in number of frames.
    vid: cv2.VideoCapture instance with the vid
    length: int, number of frames the vid must have
    """
    return int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) >= length


def validate_correctness(vid):
    #check if the video is indeed of a finger right in the camera lens
    return True

def validate_fps(vid, fps):
    #required for the frequency calculations of the bpm(heart rate)
    return vid.get(cv2.CAP_PROP_FPS) 

def process_video(video, length=600, width=320, height=240):
    """
    preprocessing the video.
    
    video: cv2.VideoCapture instance with the video
    length: number of frames to take from the cv2.VideoCapture
    width: width of the resized video
    height: height of the resized video
    """
    frames = np.zeros((length, height, width, 3), dtype=np.uint8)
    for i in range(length):
        _, frame = video.read()
        frame = cv2.resize(frame, (width, height))
        frames[i] = frame
    video.release()
    cv2.destroyAllWindows()
    return frames

def function_per_frame(video, channel, function):
    reshaped_video = video[:, :, :, channel].reshape(video.shape[0], -1)
    return function(reshaped_video, axis=-1)

def calculate_spo2(video, A=100, B=5, use_reduce=True, use_area=90, discretize=False, verbose=False):
    """
    Calculates the blood oxygen saturation.

    Algorithm based on the paper: 
    Determination of SpO2 and Heart-rate using Smartphone Camera, Kanva et al.
    https://www.iiitd.edu.in/noc/wp-content/uploads/2017/11/06959086.pdf

    video: numpy array
    A: hyperparameter used to adjust the data
    B: hyperparameter used to adjust the data
    use_reduce: if True returns a scalar, indicating the average SpO2 across the video frames
    use_area: when reducing, determines the distribution area to use to calculate the mean.
              used for robustness against outliers.
    """
    channels = {'b': 0, 'g': 1, 'r': 2}
    red_means = function_per_frame(video, channel=channels['r'], function=np.mean)
    blue_means = function_per_frame(video, channel=channels['b'], function=np.mean)
    red_stds = function_per_frame(video, channel=channels['r'], function=np.std)
    blue_stds = function_per_frame(video, channel=channels['b'], function=np.std)
    spo2 = A - B * ((red_stds / red_means) / (blue_stds / blue_means))
    if use_reduce:
        nans = np.isnan(spo2).sum()
        if verbose:
            print(f'Found {nans} NaNs, removing them')
        spo2 = spo2[~np.isnan(spo2)]
        significance = (100-use_area)/2
        lower_bound = np.percentile(spo2, significance)
        upper_bound = np.percentile(spo2, use_area + significance)
        if verbose:
            print(f'Calculating trimmed mean in the range ({lower_bound}, {upper_bound})')
        clipped_spo2 = spo2[(spo2 > lower_bound) & (spo2 < upper_bound)]
        if verbose:
            print(f'Clipped SpO2 contains {len(clipped_spo2)} values')
        mean_spo2 = int(np.mean(clipped_spo2))
        if discretize:
            return discretize_spo2(mean_spo2)
        else:
            return mean_spo2
    return spo2


def calculate_heart_rate(video, use_reduce=True, frequency_range=(20, 200), fps=30, discretize=False):
    channels = {'b': 0, 'g': 1, 'r': 2}
    lower_bound, upper_bound = frequency_range
    duration = video.shape[0] / 30
    red_means = function_per_frame(video, channel=channels['r'], function=np.mean)
    if use_reduce:
        fourier_coefs = abs(np.fft.fft(red_means)[lower_bound:upper_bound + 1])
        dominant_frequency = np.argmax(fourier_coefs) + lower_bound
        heart_rate = int(dominant_frequency * 60 / duration)
        if discretize:
            return discretize_heart_rate(heart_rate)
        else:
            return heart_rate
    return red_means


def discretize_spo2(spo2):
    if spo2 >= 95:
        return 'normal'
    elif spo2 >= 91:
        return 'hipoxia leve'
    elif spo2 >= 86:
        return 'hipoxia moderada'
    else:
        return 'hipoxia severa'


def discretize_heart_rate(bpm, age=None, sex=None, fitness_level=None):
    return bpm

file = 'vid.mp4'
video = cv2.VideoCapture(file)
validate_video(video)
video = process_video(video)

spo2 = calculate_spo2(video)
spo2_disc = calculate_spo2(video, discretize=True)

print(spo2)
print(spo2_disc)

bpm = calculate_heart_rate(video)
bpm_disc = calculate_heart_rate(video, discretize=True)

print(bpm)
print(bpm_disc)
