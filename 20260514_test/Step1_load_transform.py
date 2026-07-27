import ABRpresto
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
import pandas
import numpy as np
import time

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

def epochs_drop_unused_idxs_trials(epochs):
    # Checks th df index, drops unecessary indexes, and drops empty trials
    # Check that dataframe has the right indexes, drop unneeded indexes
    assert all([name in epochs.index.names for name in ['polarity', 'level']]), \
        'epochs dataframe must have "polarity" and level" as indexes'
    drop_these = [name for name in epochs.index.names if name not in ['polarity', 'level']]
    epochs.reset_index(drop_these, drop=True, inplace=True)
    epochs.index = epochs.index.reorder_levels(['polarity', 'level'])
    epochs.sort_index(inplace=True)

    # Drop empty trials
    all_values_0 = np.all(epochs == 0, axis=1)
    dropped_all = False
    if any(all_values_0):
        dropped_all = all(all_values_0)
        log.warning(f'\n {all_values_0.sum()}/{len(all_values_0)} epochs in this dataset have values of all 0.'
                    f' Dropping these epochs.')
        epochs = epochs[~all_values_0]
    if dropped_all:
        epoch_status = 'fail_dropped_all'
    else:
        epoch_status = 'pass'
    return epochs, epoch_status
def fs_from_timepoints(epochs):
    # If you don't have the sampling frequency you can use this to calculate it
    # from the sample times.
    # Diffs between timepoints, and how many counts of each unique tdiff
    t_diffs, tdiff_counts = np.unique(np.diff(epochs.keys().values), return_counts=True)
    fs = 1 / t_diffs[tdiff_counts.argmax()]
    return fs
def N_polarity_check(epochs, N_subaverages=2):
    N_by_polarity = epochs.groupby(['polarity', 'level']).size()
    if len(N_by_polarity) == 0:
        raise RuntimeError('N_by_polarity is empty.')
    else:
        N_min_global = N_by_polarity.min()
    if N_min_global < 100:
        msg = f'There is at least one condition with only {N_min_global} reps. Minimum needed (arbitrary) is 100.' \
              f'There are {len(epochs)} total reps for this recording'
        log.warning(msg)
        epoch_status = 'fail_insufficient_trials'
    else:
        epoch_status = 'pass'
    N_varies = len(N_by_polarity.unstack('polarity').min(axis=1).unique()) > 1
    if N_varies:
        log.warning('\n Different level combinations have different number of reps. '
                    'The number of reps per subaverage will differ by level.')
        N_per_group = 'varies'
    else:
        N_per_group = int(np.floor(N_min_global / N_subaverages))
    return epoch_status, N_per_group
# def time_window_calc(epochs, pst_range=None):
#     # Calculate time window to compute correlation over
#     if pst_range is None:
#         N_time = len(epochs.keys())
#         time_inds = np.full(N_time, True)
#     else:
#         time_inds = (epochs.keys().values >= pst_range[0]) & (
#                 epochs.keys().values < pst_range[1])
#         N_time = time_inds.sum()
#     return N_time, time_inds
#


## ABRpresto parameters
peak_lag_threshold = 0.5    # Still don't really know what this means tbh
second_filter = 'pre-average'
N_subaverages = 2

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

# Load our data
big_triggered_ad = pandas.read_pickle(
    os.path.join(output_directory, 'big_triggered_ad'))
big_triggered_neural = pandas.read_pickle(
    os.path.join(output_directory, 'big_triggered_neural'))
big_click_params = pandas.read_pickle(
    os.path.join(output_directory, 'big_click_params'))
formatted_ABR = rodgerslab_to_ABRpresto_format(big_triggered_neural, 16000)
copied_ABR = formatted_ABR.copy()
print('Yay!')

# Load example data
filename = os.path.realpath('../example_data/Example_1.csv')
print(f'Loading {filename}')
example_ABR = pandas.read_csv(filename, index_col=[0, 1, 2])
example_ABR.columns.name = 'time'
example_ABR.columns = example_ABR.columns.astype('float')

example_ABR, example_ABR_status = epochs_drop_unused_idxs_trials(example_ABR)
formatted_ABR, formatted_ABR_status = epochs_drop_unused_idxs_trials(formatted_ABR)

example_fs = fs_from_timepoints(example_ABR)
ABR_fs = fs_from_timepoints(formatted_ABR)

example_N_time, example_time_inds = N_polarity_check(example_ABR, 2)

# Set variables that will be function arguments in the future
epochs = example_ABR
fs = example_fs
second_filter_settings={'highpass': 300, 'lowpass': 3000, 'order': 1}

# filter data if requested
if second_filter == 'pre-average':
    epochs[:] = ABRpresto.utils.filter(epochs.values, fs, **second_filter_settings)

# Count number of levels, polarities, trials per rep. Calculate fs (sampling frequency) for data)
levels = epochs.index.get_level_values('level').unique().values
polarities = epochs.index.get_level_values('polarity').unique().values

peak_lag_threshold_samples = np.ceil(peak_lag_threshold * fs / 1000)

for i, level in enumerate(levels):
    t0 = time.time()
    # initialize variables for this level
    N_by_polarity = epochs.groupby(['polarity', 'level']).size()
    N_min = N_by_polarity.loc[:, level].min()
    N_per_group = int(np.floor(N_min / N_subaverages))

    # epochs_i = epochs.loc[:, :, level, :]
    # indP = epochs.loc[:, 1, level, :].index.get_level_values(0).values
    # indN = epochs.loc[:, -1, level, :].index.get_level_values(0).values
    epochsP = epochs.loc[(1, level), time_inds]
    epochsN = epochs.loc[(-1, level), time_inds]
    indP = np.arange(len(epochsP))
    indN = np.arange(len(epochsN))
    if keep_ind_masks is not None:
        indP = indP[keep_ind_masks[0]]
        indN = indN[keep_ind_masks[1]]
        N_per_group = int(min((len(indP), len(indN))) / N_subaverages)
    randomized_indicesP = np.zeros((N_per_group * N_subaverages, N_shuffles), dtype=int)
    randomized_indicesN = np.zeros((N_per_group * N_subaverages, N_shuffles), dtype=int)

    # loop across number of shuffles
    for ishuf in range(N_shuffles):
        # randomize into two buckets, evenly splitting by polarity
        randomized_indicesP[:, ishuf] = rn.choice(indP, N_per_group * N_subaverages, replace=False)
        randomized_indicesN[:, ishuf] = rn.choice(indN, N_per_group * N_subaverages, replace=False)

        # average (either mean or median)
        if avmode == 'mean':
            epochs_mean0 = epochsP.iloc[randomized_indicesP[:N_per_group, ishuf]].mean() + \
                           epochsN.iloc[randomized_indicesN[:N_per_group, ishuf]].mean()
            epochs_mean1 = epochsP.iloc[randomized_indicesP[N_per_group:, ishuf]].mean() + \
                           epochsN.iloc[randomized_indicesN[N_per_group:, ishuf]].mean()
        elif avmode == 'median':
            epochs_mean0 = np.median(np.vstack((epochsP.iloc[randomized_indicesP[:N_per_group, ishuf]].values,
                epochsN.iloc[randomized_indicesN[:N_per_group, ishuf]].values)),
                axis=0)
            epochs_mean1 = np.median(np.vstack((epochsP.iloc[randomized_indicesP[N_per_group:, ishuf]].values,
                epochsN.iloc[randomized_indicesN[N_per_group:, ishuf]].values)),
                axis=0)

        else:
            raise RuntimeError(f'Invalid avmode: {avmode}')
        # filter data if requested
        if second_filter == 'post-average':
            epochs_mean0 = utils.filter(epochs_mean0, fs, **second_filter_settings)
            epochs_mean1 = utils.filter(epochs_mean1, fs, **second_filter_settings)
        # calculate cross correlation (for 0 lag only if that's all that's needed)
        if calc_XC0m_only:
            xc0[i, ishuf] = np.corrcoef(epochs_mean0, epochs_mean1)[0, 1]
        else:
            xc[i, ishuf, :] = utils.crossCorr(epochs_mean0, epochs_mean1, norm=True)
    if not calc_XC0m_only:
        peak_index[i, :] = np.argmax(xc[i, :, :], axis=-1)
        peak_lag[i, :] = peak_index[i, :] - (N_time - 1) / 2
        peak[i, :] = np.max(xc[i, :, :], axis=-1)

    # calculate mean and std of waveform for this level
    epochs_means[:, i, 0] = epochs.xs(level, level='level').values.mean(axis=0)
    epochs_sems[:, i] = np.std(epochs.xs(level, level='level').values, axis=0, ddof=1) / \
                        np.sqrt(len(indP) + len(indN))
    # find which shuffle yielded a cross-corraltion value closest to the mean
    if calc_XC0m_only:
        ishuf_mean = np.argmin(np.abs(xc0[i, :] - xc0[i, :].mean()))
    else:
        ishuf_mean = np.argmin(np.abs(xc[i, :, L0] - xc[i, :, L0].mean()))
    # Create and save subaverages for this shuffle to show on plot later
    if avmode == 'mean':
        epochs_means[:, i, 1] = np.mean(
            np.vstack((epochs.loc[(1, level)].iloc[randomized_indicesP[:N_per_group, ishuf_mean]].values,
                epochs.loc[(-1, level)].iloc[randomized_indicesN[:N_per_group, ishuf_mean]].values)),
            axis=0)
        epochs_means[:, i, 2] = np.mean(
            np.vstack((epochs.loc[(1, level)].iloc[randomized_indicesP[N_per_group:, ishuf_mean]].values,
                epochs.loc[(-1, level)].iloc[randomized_indicesN[N_per_group:, ishuf_mean]].values)),
            axis=0)
    elif avmode == 'median':
        epochs_means[:, i, 1] = np.median(
            np.vstack((epochs.loc[(1, level)].iloc[randomized_indicesP[:N_per_group, ishuf_mean]].values,
                epochs.loc[(-1, level)].iloc[randomized_indicesN[:N_per_group, ishuf_mean]].values)),
            axis=0)
        epochs_means[:, i, 2] = np.median(
            np.vstack((epochs.loc[(1, level)].iloc[randomized_indicesP[N_per_group:, ishuf_mean]].values,
                epochs.loc[(-1, level)].iloc[randomized_indicesN[N_per_group:, ishuf_mean]].values)),
            axis=0)
    if second_filter == 'post-average':
        epochs_means[:, i, 0] = utils.filter(epochs_means[:, i, 0], fs, **second_filter_settings)
        epochs_means[:, i, 1] = utils.filter(epochs_means[:, i, 1], fs, **second_filter_settings)
        epochs_means[:, i, 2] = utils.filter(epochs_means[:, i, 2], fs, **second_filter_settings)
    t1 = time.time()
    comptime[i] = t1 - t0


