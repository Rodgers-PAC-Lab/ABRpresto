import ABRpresto
import os
import matplotlib
matplotlib.use('TkAgg')
import logging
import matplotlib.pyplot as plt
import numpy as np
# from . import utils
import scipy.stats
import time
import pandas
import warnings
import json

log = logging.getLogger(__name__)
colors = plt.get_cmap('tab10').colors

def plot_fit(levels, levels_, xc0, ABRtime, epochs_means, epochs_sems, pst_range, fit_XC0m,
             norm_waveforms=True, human_threshold=None, avmode='mean', criterion=0.3):
    # In the left column the figures show mean +/- SE of all trials in black, and median (or mean, depending on AVmode)
    # for the two subsets. Waveforms are normalized (for each level all 3 lines are scaled by the peak-to-peak of
    # the mean of all trials).
    # The right hand side shows mean correlation coefficient vs stimulus level. Sigmoid and
    # power law fits to this data are shown in green and purple. The threshold is shown by the pink dashed line.

    fs_scale = 1
    fs_labels = 10*fs_scale
    fs_ticklabels = 10*fs_scale

    udiffs, counts = np.unique(np.diff(np.array(levels)), return_counts=True)
    m = counts.argmax()
    level_diff_mode = udiffs[m]

    if pst_range is not None:
        ii = (ABRtime > pst_range[0]*1000) & (ABRtime <= pst_range[1]*1000)
    else:
        ii = np.full(ABRtime.shape, True)

    fig_handle, ax = plt.subplots(1, 2, figsize=(7, 10), gridspec_kw={'hspace': 0.07, 'wspace': .25, 'top': 1, 'bottom': 0.07,
                                                               'left': .07, 'right': 1, 'width_ratios': [.4, .6]})
    for level in levels_:
        i = np.where(level == levels)[0][0]
        if epochs_means is not None:
            y = epochs_means[:, i, 0].T * 1e6  # normalizing to full waveform

            if norm_waveforms:
                ys = 1 / (y[ii].max(axis=0) - y[ii].min(axis=0))
            else:
                ys = 1
            # plotting just windowed data
            h0 = ax[0].plot(ABRtime[ii], y[ii] * ys * level_diff_mode + level, 'k', linewidth=1)

            h0f = ax[0].fill_between(ABRtime[ii],
                        np.squeeze(epochs_means[ii, i, 0] - epochs_sems[ii, i]) * 1e6 * ys * level_diff_mode + level,
                        np.squeeze(epochs_means[ii, i, 0] + epochs_sems[ii, i]) * 1e6 * ys * level_diff_mode + level,
                        color='lightgrey', alpha=.5)

            h1 = ax[0].plot(ABRtime[ii], epochs_means[ii, i, 1] * 1e6 * ys * level_diff_mode + level, color=colors[0],
                       linewidth=1)
            h2 = ax[0].plot(ABRtime[ii], epochs_means[ii, i, 2] * 1e6 * ys * level_diff_mode + level, color=colors[1],
                       linewidth=1)
            if pst_range is not None:
                ax[0].set_xlim(np.array(pst_range) * 1000)
    if norm_waveforms:
        ax[0].set_ylim((levels_[0]-level_diff_mode*.7, levels_[-1]+ 0.7*level_diff_mode + 0.09*(levels_[-1]-levels_[0])))
    ax[0].set_xlabel('Time (ms)', fontsize=fs_labels)
    ax[0].set_ylabel('Level (dB SPL)', fontsize=fs_labels)
    ax[0].legend([(h0[0], h0f), h1[0], h2[0]], ['mean \u00B1 SE of all trials', avmode +' of 1st subset',
                                         avmode +' of 2nd subset'], loc='upper left', bbox_to_anchor=(0,1))
    ax[0].set_yticks(levels_)
    time_lines = np.array([2, 4, 6, 8])
    if pst_range is not None:
        time_lines = time_lines[(time_lines > pst_range[0]*1000) & (time_lines < pst_range[1]*1000)]
    for tl in time_lines:
        ax[0].axvline(tl, color='lightgrey', zorder=-10, linewidth=.5)

    # ax[0].set_ylabel('Level (dB SPL)')

    if fit_XC0m is not None:
        # levi = [i for i,lev in enumerate(levels) if any(lev==np.array([30,65]))]
        # h = ax[1].violinplot(xc0[levi,:].T, levels[levi], widths=np.diff(levels[:2])[0]*.9, showextrema=False, points=200,
        #                    showmeans=False)
        h = ax[1].violinplot(xc0.T, levels, widths=np.diff(levels[:2])[0]*.9, showextrema=False, points=200,
                           showmeans=False)
        for pc in h['bodies']:
            pc.set_facecolor('lightgrey')
            # pc.set_edgecolor('black')
            pc.set_alpha(.5)
        ax[1].plot(levels, xc0.mean(axis=1), '.k', label='data mean', )
        # ax[1].plot(levels[levi], xc0[levi,:].mean(axis=1), '.k', label='data mean', )
        ax[1].set_ylim((-.3, 1.01))
        yf = None
        if fit_XC0m['sigmoid_fit'] is not None:
            if fit_XC0m['bestFitType'] == 'sigmoid':
                l = f"sigmoid fit,\nthresh={fit_XC0m['threshold']:.1f}"
                lw = 2
                yf = fit_XC0m['sigmoid_fit']['yfit']
            else:
                l = f"sigmoid fit,\nnot used"
                lw = 1

            ax[1].plot(levels, fit_XC0m['sigmoid_fit']['yfit'], color=colors[2], lw=lw, label=l)
            if fit_XC0m['bestFitType'] == 'power law':
                l = f"power law fit,\nthresh={fit_XC0m['threshold']:.1f}"
                lw = 2
                ls = '-'
                yf = fit_XC0m['power_law_fit']['yfit']
            elif fit_XC0m['bestFitType'] == 'power law (noisy)':
                l = f"Noisy power law fit,\nthresh={fit_XC0m['threshold']:.1f}"
                lw = 2
                ls = '--'
                yf = fit_XC0m['power_law_fit']['yfit']
            else:
                l = f"power law fit,\nnot used"
                lw = 1
                ls = '-'
            ax[1].plot(levels, fit_XC0m['power_law_fit']['yfit'], color=colors[4], lw=lw, label=l, ls=ls)

        if fit_XC0m['threshold'] is not None:
            if fit_XC0m['threshold'] is np.inf:
                l = 'thresh=inf\n(all levels below criterion)'
            elif np.isnan(fit_XC0m['threshold']):
                l = 'thresh is nan\n(error fitting?)'
            else:
                l = None
            ax[1].axvline(fit_XC0m['threshold'], color=colors[6], linestyle='--', linewidth=1, label=l)
            time_range = np.array((ABRtime[ii][0], ABRtime[ii][-1]))
            ax[0].plot((time_range[0]-np.diff(time_range)*.25, time_range[1]+np.diff(time_range)*.05),
                       fit_XC0m['threshold']*np.ones(2), color=colors[6], linestyle='--', linewidth=1)
            ax[0].set_clip_on(False)
        ax[1].axhline(criterion, color='k', linestyle='--', linewidth=.5)
        ax[1].text(ax[1].get_xlim()[1], criterion, 'criterion', horizontalalignment='right', fontsize=10*fs_scale)
        ax[1].set_ylabel('Correlation Coefficient', fontsize=fs_labels)
        ax[1].set_yticks((0, .2, .4, .6, .8, 1))
        ax[1].set_yticklabels(('0', '.2', '.4', '.6', '.8', '1'))
        ax[1].set_xlabel('Level (dB SPL)', fontsize=fs_labels)
        levi = (np.abs(levels-np.percentile(ax[1].get_xlim(),35))).argmin()
        if (yf is None) or yf[levi] < .8:
            ax[1].legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=True, fontsize=10 * fs_scale, framealpha=1)
        else:
            ax[1].legend(loc='lower right', bbox_to_anchor=(1, 0), frameon=True, fontsize=10 * fs_scale, framealpha=1)
        ax[0].tick_params(labelsize=fs_ticklabels)
        ax[1].tick_params(labelsize=fs_ticklabels)
        ax[1].set_clip_on(False)
    # Hide the right and top spines
    [ax_.spines.right.set_visible(False) for ax_ in ax]
    [ax_.spines.top.set_visible(False) for ax_ in ax]

    if human_threshold is not None:
        ax[1].axvline(human_threshold, color='k', linestyle='--', linewidth=1)
    return fig_handle
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

RV_channel = big_triggered_neural.xs('RV', level='channel')

XCsubargs = {
    'seed': 0,
    'pst_range': [0.0005, 0.006],
    'N_shuffles': 500,
    'avmode': 'median',
    'peak_lag_threshold': 0.5,
    'XC0m_threshold': 0.3,
    'XC0m_sigbounds': 'increasing, midpoint within one step of x range',  # sets bounds to make slope positive,
    # and midpoint within [min(level) - step, max(level) + step] step is usually 5 dB
    'XC0m_plbounds': 'increasing',  # sets bounds to make slope positive
    'second_filter': 'pre-average',
    'calc_XC0m_only': True,
    'save_data_resamples': False  # use this to save intermediate data (from each resample)
}

print(os.path.basename(__file__))

filename = os.path.realpath('../example_data/Example_1.csv')
print(f'Loading {filename}')
example_abr_single_trial_data = pandas.read_csv(filename, index_col=[0, 1, 2])
example_abr_single_trial_data.columns.name = 'time'
example_abr_single_trial_data.columns = example_abr_single_trial_data.columns.astype('float')

abr_single_trial_data = -RV_channel
abr_single_trial_data.columns.name = 'time'
abr_single_trial_data.columns = abr_single_trial_data.columns.astype('float')
abr_single_trial_data.index = abr_single_trial_data.index.rename("level", level=1)

# Convert sample number to time in s
# They calculate sampling frequency from the interval between column values
sampling_rate = 16000
t = abr_single_trial_data.columns / sampling_rate
abr_single_trial_data.columns = t.values

# Their polarity values are 1 and -1 instead of True and False, so change ours to match.
abr_single_trial_data = abr_single_trial_data.reset_index()
abr_single_trial_data['polarity'] = abr_single_trial_data['polarity'].map({
    True: 1, False: -1})
abr_single_trial_data = abr_single_trial_data.set_index(
    ['recording','level','polarity','t_samples'])
copied = abr_single_trial_data.copy()
# plt.plot(abr_single_trial_data[0:3].values[0:3].T)
print('Fitting with ABRpresto algorithm')

levels = example_abr_single_trial_data.index.get_level_values('level').unique()
fs_scale = 1
fs_labels = 10 * fs_scale
fs_ticklabels = 10 * fs_scale

udiffs, counts = np.unique(np.diff(np.array(levels)), return_counts=True)
m = counts.argmax()
level_diff_mode = udiffs[m]

epochs = example_abr_single_trial_data
# epochs_means = np.zeros((len(epochs.keys()), len(levels)))

fit_results, fig_handle, epochs_means = ABRpresto.XCsub.estimate_threshold_wmeans(abr_single_trial_data, **XCsubargs)
fit_results['ABRpresto version'] = ABRpresto.get_version()

fit_results_ex, fig_handle_ex, epochs_means_ex = ABRpresto.XCsub.estimate_threshold_wmeans(example_abr_single_trial_data, **XCsubargs)
fit_results_ex['ABRpresto version'] = ABRpresto.get_version()
our_ts = abr_single_trial_data.columns
their_ts = example_abr_single_trial_data.columns
our_ts_shifted = our_ts + 0.55/1000
# plot_dict = {
#     'ours_normal' : [our_ts,epochs_means[:,10,0], epochs_means[:,10,0].max() - epochs_means[:,10,0].min()]
#     'theirs_normal': [their_ts, epochs_means_ex[:, 10, 0], epochs_means_ex[:, 10, 0].max() - epochs_means_ex[:, 10, 0].min()]
#     'ours_normal': [our_ts_shifted, epochs_means[:, 10, 0], epochs_means[:, 10, 0].max() - epochs_means[:, 10, 0].min()]
# }
our_range = epochs_means[:,10,0].max() - epochs_means[:,10,0].min()
their_range = epochs_means_ex[:, 10, 0].max() - epochs_means_ex[:, 10, 0].min()

f, ax = plt.subplots()
ax.plot(our_ts,epochs_means[:,10,0])
ax.plot(their_ts, epochs_means_ex[:,10,0])
ax.plot(our_ts_shifted,epochs_means[:,10,0])

norm_waveforms = True
f, axa = plt.subplots(1,2)
axa[0].plot(our_ts,epochs_means[:,10,0], label='Lighthouse_232 RV')
axa[0].plot(their_ts, epochs_means_ex[:,10,0] * our_range/their_range, label='example_1')
axa[1].plot(our_ts_shifted,epochs_means[:,10,0], label='Lighthouse_232 RV')
axa[1].plot(their_ts, epochs_means_ex[:,10,0] * our_range/their_range, label='example_1')
for ax in axa:
    ax.set_xlabel('time (s)')
    ax.set_ylabel('ABR normalized')
axa[0].set_title('Actual times')
axa[1].set_title('Our ABR shifted +0.55 ms')
# ABRtime = epochs.keys().values.astype('float') * 1000
# pst_range = [0.0005, 0.006]
# ii = (ABRtime > pst_range[0]*1000) & (ABRtime <= pst_range[1]*1000)
# udiffs, counts = np.unique(np.diff(np.array(levels)), return_counts=True)
# m = counts.argmax()
# level_diff_mode = udiffs[m]

# f,ax = plt.subplots()
# for i, level in enumerate(levels):
#     epochs_means[:, i] = epochs.xs(level, level='level').values.mean(axis=0)
#     y = epochs_means[:, i].T * 1e6
#     if norm_waveforms:
#         ys = 1 / (y[ii].max(axis=0) - y[ii].min(axis=0))
#     else:
#         ys = 1
#     # plotting just windowed data
#     h0 = ax.plot(ABRtime[ii], y[ii] * ys * level_diff_mode + level, 'k', linewidth=1)
#
# example_65 = example_abr_single_trial_data.xs(65, level='level')
# Lighthouse232_65 = abr_single_trial_data.xs(65, level='level')
# Lighthouse232_65_shifted = Lighthouse232_65.copy()
# # Try it with their normalization
# f,ax = plt.subplots()
# for epochs in [epochs_means[:,10,0],epochs_means_ex[:,10,0]]:
#     ABRtime = epochs.keys().values.astype('float') * 1000
#     ii = (ABRtime > pst_range[0] * 1000) & (ABRtime <= pst_range[1] * 1000)
#     y = epochs.mean() * 1e6
#     if norm_waveforms:
#         ys = 1 / (y[ii].max(axis=0) - y[ii].min(axis=0))
#     else:
#         ys = 1
#     h0 = ax.plot(ABRtime[ii], y[ii] * ys, linewidth=1)
