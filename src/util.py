import numpy as np
import pandas as pd

from sklearn import metrics


def print_time_taken(time_taken):

    h,m = divmod(time_taken,60*60)
    m,s = divmod(m,60)
    time_taken = f"{h:.0f} h {m:.0f} min {s:.2f} sec" if h > 0 else f"{m:.0f} min {s:.2f} sec"
    time_taken = f"{m:.0f} min {s:.2f} sec" if m > 0 else f"{s:.2f} sec"

    print(f"\nTime Taken: {time_taken}")
