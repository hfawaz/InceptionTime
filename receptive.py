import numpy as np
import sklearn
import pandas as pd
import sys
from classifiers.inception import Classifier_INCEPTION
import subprocess
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

from utils.utils import transform_labels
from utils.utils import create_synthetic_dataset
from utils.utils import create_directory
from utils.utils import check_if_file_exits
from utils.utils import generate_array_of_colors

import matplotlib

import matplotlib.pyplot as plt

root_dir = '/b/home/uha/hfawaz-datas/temp-dl-tsc/'
root_output_directory = root_dir + 'receptive-field/exp/'
root_output_directory_df = root_dir + 'receptive-field/'
create_directory(root_output_directory_df)

BATCH_SIZE = 128
NB_EPOCHS = 500

pattern_lens = [[0.1]]

pattern_poss = [[0.1, 0.65]]
# uncomment the following to experiment with the number of classes in a dataset
# pattern_poss = [
#     [0.1, 0.65],
#     [0.1, 0.3, 0.65],
#     [0.1, 0.3, 0.5, 0.65],
#     [0.1, 0.3, 0.5, 0.7, 0.9],
#     [0.1, 0.2, 0.3, 0.4, 0.5, 0.65],
#     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
#     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
#     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# ]

kernel_sizes = np.arange(start=6, stop=15, step=2 * 4)  # default 41

depths = np.arange(start=1, stop=10, step=1)  # default 6

filters_s = [1 * (2 ** i) for i in range(8)]  # default 32

ts_lens = [16 * (2 ** i) for i in range(7)]  # default 256

ts_ns = [6 * (2 ** i) for i in range(8)]  # default 128

use_residuals = [True, False]  # default true

use_bottlenecks = [True, False]  # default true


def convert_to_float(s):
    s = s.split('_')[0]
    return float(s)


subprocess.call(['./receptive_field_remove_non_completed.sh'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if len(sys.argv) == 1:
    matplotlib.use('agg')
    out_df = root_output_directory_df + 'df_res_sub_0.csv'

    idx_out = 1
    while check_if_file_exits(out_df):
        out_df = root_output_directory_df + 'df_res_sub_' + str(idx_out) + '.csv'
        idx_out += 1

    columns = ['pattern_len', 'pattern_pos', 'ts_len', 'ts_n', 'nb_classes',
               'filters', 'kernel_size', 'depth', 'use_residual', 'use_bottleneck', 'accuracy']

    df_results = pd.DataFrame(index=[], columns=columns)

    df_results.to_csv(out_df)

    curr_idx = 0

    for pattern_len_ in pattern_lens:

        for pattern_pos in pattern_poss:

            pattern_len = pattern_len_

            nb_classes = len(pattern_len) * len(pattern_pos)

            for ts_len in ts_lens:

                for ts_n in ts_ns:

                    x_train, y_train, x_test, y_test = create_synthetic_dataset(pattern_len=pattern_len,
                                                                                pattern_pos=pattern_pos,
                                                                                ts_len=ts_len, ts_n=ts_n)

                    # make the min to zero of labels
                    y_train, y_test = transform_labels(y_train, y_test)

                    # save orignal y because later we will use binary
                    y_true = y_test.astype(np.int64)
                    y_true_train = y_train.astype(np.int64)
                    # transform the labels from integers to one hot vectors
                    enc = sklearn.preprocessing.OneHotEncoder()
                    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
                    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
                    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

                    if len(x_train.shape) == 2:
                        # if uni-variate add a dimension to make it multivariate with one dimension
                        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
                        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

                    input_shape = x_train.shape[1:]

                    for kernel_size in kernel_sizes:

                        for depth in depths:

                            for filters in filters_s:

                                for use_residual in use_residuals:

                                    for use_bottleneck in use_bottlenecks:

                                        output_directory = root_output_directory + 'pattern_len_' + ''.join(
                                            [str(e) + '_' for e in pattern_len]) + '/'
                                        output_directory = output_directory + 'pattern_pos_' + ''.join(
                                            [str(e) + '_' for e in pattern_pos]) + '/'
                                        output_directory = output_directory + 'ts_len_' + str(ts_len) + '/'
                                        output_directory = output_directory + 'ts_n_' + str(ts_n) + '/'
                                        output_directory = output_directory + 'nb_classes_' + str(nb_classes) + '/'
                                        output_directory = output_directory + 'filters_' + str(filters) + '/'
                                        output_directory = output_directory + 'kernel_size_' + str(kernel_size) + '/'
                                        output_directory = output_directory + 'depth_' + str(depth) + '/'
                                        output_directory = output_directory + 'use_residual_' + str(use_residual) + '/'
                                        output_directory = output_directory + 'use_bottleneck_' + str(
                                            use_bottleneck) + '/'

                                        test_dir = create_directory(output_directory)

                                        if test_dir is None:
                                            # job already done / is being done
                                            continue

                                        # create the classifier
                                        clf = Classifier_INCEPTION(output_directory, input_shape, nb_classes,
                                                                   nb_filters=filters, use_residual=use_residual,
                                                                   use_bottleneck=use_bottleneck, depth=depth,
                                                                   kernel_size=int(kernel_size), verbose=False,
                                                                   batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS)
                                        # train
                                        df_metrics = clf.fit(x_train, y_train, x_test, y_test, y_true)

                                        acc = df_metrics['accuracy'].values[0]

                                        df_results.loc[curr_idx] = [''.join([str(e) + '_' for e in pattern_len]),
                                                                    ''.join([str(e) + '_' for e in pattern_pos]),
                                                                    ts_len, ts_n,
                                                                    nb_classes,
                                                                    filters, kernel_size, depth, use_residual,
                                                                    use_bottleneck,
                                                                    acc]

                                        curr_idx += 1

                                        df_results.to_csv(out_df)

    print('DONE')

elif sys.argv[1] == 'plot_results':
    matplotlib.use('pdf')
    plot_3d = False

    # read results
    df_results = pd.read_csv(root_output_directory_df + 'df_res_sub_0.csv', index_col=0)

    out_df = root_output_directory_df + 'df_res_sub_1.csv'

    out_df_idx = 2
    while check_if_file_exits(out_df):
        df_results_temp = pd.read_csv(out_df, index_col=0)
        df_results = pd.concat([df_results, df_results_temp], sort=True)
        out_df = root_output_directory_df + 'df_res_sub_' + str(out_df_idx) + '.csv'
        out_df_idx += 1

    df_results.reset_index(inplace=True)

    split_ons = ['use_bottleneck', 'use_residual', 'ts_n',
                 'filters', 'nb_classes']
    split_values = [True, True, 128, 32, 2]

    x_label = 'ts_len'
    y_label = 'RF'
    z_label = 'accuracy'

    # add the receptive field value
    df_results['RF'] = (df_results['kernel_size'] - 1) * df_results['depth'] + 1

    curr_df = df_results

    for i in range(len(split_values)):
        split_on = split_ons[i]
        split_value = split_values[i]
        curr_df = curr_df.loc[df_results[split_on] == split_value]

    x_labels = pd.unique(curr_df[x_label])

    colors = generate_array_of_colors(len(x_labels))

    fig = plt.figure()

    i = 0

    for xx in x_labels:
        curr_df_p = curr_df.loc[curr_df[x_label] == xx].sort_values([y_label])

        ts = pd.Series(index=curr_df_p[y_label], data=curr_df_p[z_label].values)
        ts = ts.rolling(window=3, min_periods=1).mean()
        ts.plot(color=colors[i], label=x_label + str(xx))

        i += 1

    plt.xlabel(y_label)
    plt.ylabel(z_label)

    plt.legend(loc='best')

    plt.savefig(root_output_directory_df + 'out.pdf')
    plt.close()
