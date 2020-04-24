# Delay line detector codes

import numpy as np

def cfd(x, y, shift=0.008, threshold=0.015, deadtime=0.01):
    # Simple Constant Fraction Discriminator for hit finding

    pixel_shift = int(shift / np.diff(x).mean() / 2)
    y1, y2 = y[:-2*pixel_shift], y[2*pixel_shift:]
    x_, y_ = x[pixel_shift:-pixel_shift], y[pixel_shift:-pixel_shift]
    y3 = y1 - y2
    peak_idx = np.where((y3[:-1]<0)&(0<=y3[1:])&(y_[1:]>threshold))[0]
    times, amplitudes = x_[1:][peak_idx], y_[1:][peak_idx]
    if len(times)==0:
        return [], []
    else:
        deadtime_filter = [0]
        previous_time = times[0]
        for i, time in enumerate(times[1:]):
            if time - previous_time > deadtime:
                deadtime_filter.append(i+1)
                previous_time = time
        return times[deadtime_filter], amplitudes[deadtime_filter]
