import ABRpresto
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
import pandas

def rodgerslab_to_ABRpresto_format(big_triggered_neural_df, sampling_rate=16000):
    # Converts one of our ABR data dataframes into a format ABRpresto can use.
    # ABRpresto runs on a large number of trials (500+ I think? Or maybe 5,000+)
    # before any averaging. It usually needs more than one recording
    # to get enough trials.
    # The index needs to include:
    #   'level' :       sound levels in dB, which we called 'labels'
    #   'polarity' :    speaker voltage polarity, which needs to be 1 or -1,
    #                   and ours is T/F

    # First, invert the signal to match convention
    formatted_ABR_session = -big_triggered_neural_df

    # Rename the columns as 'time'
    formatted_ABR_session.columns.name = 'time'
    formatted_ABR_session.columns = formatted_ABR_session.columns.astype('float')
    formatted_ABR_session.index = formatted_ABR_session.index.rename("level", level=1)

    # Convert sample number to time in s
    # They calculate sampling frequency from the interval between column values
    t = formatted_ABR_session.columns / sampling_rate
    formatted_ABR_session.columns = t.values

    # Their polarity values are 1 and -1 instead of True and False, so change ours to match.
    og_index = formatted_ABR_session.index.names
    formatted_ABR_session = formatted_ABR_session.reset_index()
    formatted_ABR_session['polarity'] = formatted_ABR_session['polarity'].map({
        True: 1, False: -1})
    formatted_ABR_session = formatted_ABR_session.set_index(og_index)
    return formatted_ABR_session

## Paths
# Load the required file filepaths.json (see README)
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']

# Testing on Lighthouse_232 from 2025-08-15, his first ABR.
# This is not the guy with the ear infection of course.
## Load previous results
# Load results of Step1
recording_metadata = pandas.read_pickle(
    os.path.normpath(os.path.join(output_directory, 'recording_metadata')))

# Load results of Step2
big_triggered_ad = pandas.read_pickle(
    os.path.join(output_directory, 'big_triggered_ad'))
big_triggered_neural = pandas.read_pickle(
    os.path.join(output_directory, 'big_triggered_neural'))
big_click_params = pandas.read_pickle(
    os.path.join(output_directory, 'big_click_params'))
formatted_ABR = rodgerslab_to_ABRpresto_format(big_triggered_neural, 16000)
print('Yay!')