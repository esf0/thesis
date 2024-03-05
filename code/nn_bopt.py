import os
import csv
import numpy as np
from tqdm import tqdm

from os.path import dirname, join as pjoin
import scipy as sp
import scipy.io as sio

import tensorflow as tf
from keras import callbacks
from keras.models import Sequential
from keras.layers import Activation, Dense, Reshape, Flatten, BatchNormalization, Conv1D

from skopt import gbrt_minimize, gp_minimize, load
from skopt.utils import use_named_args

from skopt.space import Real, Categorical, Integer
from skopt.callbacks import CheckpointSaver

job_name = "polycoef_a_norm_1"
# DRIVE_DIR = "/home/esf0/data_for_nn/data_nfdm/"
DRIVE_DIR = "/work/ec180/ec180/esedov/data_for_polycoef/"
checkpoint_filepath = DRIVE_DIR + 'model_' + job_name + '_checkpoint'
skopt_checkpoint_file = "b_checkpoint_" + job_name + ".pkl"
skopt_param_file = "bo_param_" + job_name + ".txt"
load_flag = True  # previous bayesian optimiser state

input_shape = (1024, 2)
output_shape = (1024, 2)

patience = 500
n_epochs = 30000
n_params_max = 20_000_000
initial_batch_size = 15000

# tf.keras.losses.MeanSquaredError()
# tf.keras.losses.MeanSquaredLogarithmicError()
loss_function = tf.keras.losses.MeanSquaredError()
learning_rate = 1e-4

f_data_load = True
if f_data_load:
    train_x = np.load(DRIVE_DIR + "data_polycoef_train_x_p_0_50_run_0.npy", allow_pickle=True)
    train_y = np.load(DRIVE_DIR + "data_polycoef_train_a_normalised_p_0_50_run_0.npy", allow_pickle=True)

    test_x = np.load(DRIVE_DIR + "data_polycoef_test_x_p_0_50_run_0.npy", allow_pickle=True)
    test_y = np.load(DRIVE_DIR + "data_polycoef_test_a_normalised_p_0_50_run_0.npy", allow_pickle=True)

    print('data loaded')


n_conv_layers = Integer(low=2, high=4, name='n_conv')

n_filters_1 = Integer(low=10, high=128, name='filters1')
n_filters_2 = Integer(low=10, high=128, name='filters2')
n_filters_3 = Integer(low=10, high=256, name='filters3')
n_filters_4 = Integer(low=10, high=256, name='filters4')

s_kernel_1 = Integer(low=1, high=20, name='kernel1')
s_kernel_2 = Integer(low=1, high=20, name='kernel2')
s_kernel_3 = Integer(low=1, high=20, name='kernel3')
s_kernel_4 = Integer(low=1, high=20, name='kernel4')

stride_1 = Integer(low=1, high=3, name='stride1')
stride_2 = Integer(low=1, high=3, name='stride2')
stride_3 = Integer(low=1, high=3, name='stride3')
stride_4 = Integer(low=1, high=3, name='stride4')

dilation_1 = Integer(low=1, high=3, name='dilation1')
dilation_2 = Integer(low=1, high=3, name='dilation2')
dilation_3 = Integer(low=1, high=3, name='dilation3')
dilation_4 = Integer(low=1, high=3, name='dilation4')

n_dense_layers = Integer(low=1, high=3, name='n_dense')
n_dense_cell_1 = Integer(low=128, high=4096, name='cell1')
n_dense_cell_2 = Integer(low=128, high=4096, name='cell2')
n_dense_cell_3 = Integer(low=128, high=4096, name='cell3')

act_func_1 = Categorical(name='activation1', categories=['relu', 'sigmoid', 'tanh'])
act_func_2 = Categorical(name='activation2', categories=['relu', 'sigmoid', 'tanh'])
act_func_3 = Categorical(name='activation3', categories=['relu', 'sigmoid', 'tanh'])
act_func_4 = Categorical(name='activation4', categories=['relu', 'sigmoid', 'tanh'])
act_func_5 = Categorical(name='activation5', categories=['relu', 'sigmoid', 'tanh'])
act_func_6 = Categorical(name='activation6', categories=['relu', 'sigmoid', 'tanh'])
act_func_7 = Categorical(name='activation7', categories=['relu', 'sigmoid', 'tanh'])


dimensions = [n_conv_layers,
              n_filters_1, n_filters_2, n_filters_3, n_filters_4,
              s_kernel_1, s_kernel_2, s_kernel_3, s_kernel_4,
              stride_1, stride_2, stride_3, stride_4,
              dilation_1, dilation_2, dilation_3, dilation_4,
              n_dense_layers,
              n_dense_cell_1, n_dense_cell_2, n_dense_cell_3,
              act_func_1, act_func_2, act_func_3, act_func_4, act_func_5, act_func_6, act_func_7]


def parse_arguments(kwargs):
    # print('kwargs: ', kwargs)
    # parse arguments for convolutional layers
    # print(kwargs['n_conv'])
    n_conv = kwargs['n_conv']

    n_filters = []
    s_kernel = []
    stride = []
    dilation = []
    for k in range(n_conv):
        # print(kwargs['filters' + str(k + 1)])
        n_filters.append(kwargs['filters' + str(k + 1)])
        dilation.append(tuple([kwargs['dilation' + str(k + 1)]]))
        s_kernel.append(tuple([kwargs['kernel' + str(k + 1)]]))
        if kwargs['dilation' + str(k + 1)] > 1:
            stride.append(tuple([1]))  # TF doesn't support stride > 1 for dilation > 1
        else:
            stride.append(tuple([kwargs['stride' + str(k + 1)]]))

    # print(n_filters, s_kernel, stride, dilation)

    # parse arguments for dense layers
    n_dense = kwargs['n_dense']
    # print(n_dense)

    n_cell = []
    for k in range(n_dense):
        n_cell.append(kwargs['cell' + str(k + 1)])

    activation = []
    for k in range(n_dense + n_conv):
        activation.append(kwargs['activation' + str(k + 1)])

    # print(n_cell, activation)

    return n_conv, n_dense, n_filters, s_kernel, stride, dilation, n_cell, activation


def create_model(n_conv, n_filters, s_kernel, stride, dilation, n_dense, n_cell, activation, padding='valid',
                 input_shape=(1024, 2), output_shape=1024):
    # build convolutional neural network
    # first layer has to have input shape
    list_of_layers = [
        Conv1D(filters=n_filters[0], kernel_size=s_kernel[0], strides=stride[0], dilation_rate=dilation[0],
               activation=activation[0], padding=padding, input_shape=input_shape)]
    for k in range(1, n_conv):
        list_of_layers.append(
            Conv1D(filters=n_filters[k], kernel_size=s_kernel[k], strides=stride[k], dilation_rate=dilation[k],
                   activation=activation[k], padding=padding))
    list_of_layers.append(Flatten())
    for k in range(n_dense):
        list_of_layers.append(Dense(units=n_cell[k], activation=activation[k + n_conv]))

    if len(np.shape(output_shape)) == 0:
        list_of_layers.append(Dense(units=output_shape))  # output layer has to have output shape
    else:
        list_of_layers.append(Dense(units=(output_shape[0] * output_shape[1])))
        list_of_layers.append(Reshape(output_shape))
    model = Sequential(list_of_layers)

    return model


@use_named_args(dimensions=dimensions)
def fitness(**kwargs):
    # this function is designed to make an arbitrary convolution neural network
    # so we have to parse argument in the beginning

    n_conv, n_dense, n_filters, s_kernel, stride, dilation, n_cell, activation = parse_arguments(kwargs)
    # print(n_conv, n_filters, s_kernel, stride, dilation)
    # print(n_dense, n_cell, activation)

    losses = np.array([999.0])
    if load_flag:
        try:
            losses = np.load(DRIVE_DIR + 'losses_' + job_name + '.npy', allow_pickle=True)
        except Exception as e:
            print("Problem with loading losses", e)
            losses = np.array([999.0])

    padding = 'valid'

    loss = 100.0

    model_build_flag = False

    # build convolutional neural network
    try:
        model = create_model(n_conv, n_filters, s_kernel, stride, dilation, n_dense, n_cell, activation,
                             padding=padding, input_shape=input_shape, output_shape=output_shape)
        model_build_flag = True

    except Exception as e:
        # print("Probably too big batch size", e)
        print("Cannot create model")

    if model_build_flag:

        model.compile(loss=loss_function,  # keras.losses.mean_squared_error
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # keras.optimizers.Adam(lr=0.001)
                      metrics=['accuracy'])

        if model.count_params() > n_params_max:

            loss = 10.0

        else:

            batch_size = initial_batch_size
            flag_success = False

            while not flag_success:

                earlystopping = callbacks.EarlyStopping(monitor="loss",
                                                        mode="auto",
                                                        patience=patience,
                                                        restore_best_weights=True)

                checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                       save_weights_only=False,
                                                       save_best_only=True,
                                                       monitor='val_loss',
                                                       mode='min')

                try:
                    history_w_aug = model.fit(train_x, train_y,
                                              validation_data=(test_x, test_y),
                                              epochs=n_epochs,
                                              batch_size=batch_size, verbose=0,
                                              callbacks=[earlystopping])
                except Exception as e:
                    print("Probably too big batch size", e)
                    batch_size = int(batch_size / 2)
                else:
                    flag_success = True

            predictions = model.predict(test_x)
            mse = tf.keras.losses.MeanSquaredError()
            loss = mse(test_y, predictions).numpy()

            # check if loss is less then other and save
            if loss < np.min(np.array(losses)):
                # save model
                model.save(DRIVE_DIR + 'model_' + job_name + '.h5')

            del model

    # save model

    # Print all the hyperparameters
    print('###########################################################')
    print('loss MSE:', loss)
    print(f"{loss} {n_conv} " +
          f"{' '.join([str(l) for l in n_filters])} " +
          f"{' '.join([str(l) for l in s_kernel])} " +
          f"{' '.join([str(l) for l in stride])} " +
          f"{' '.join([str(l) for l in dilation])} " +
          f"{n_dense} " +
          f"{' '.join([str(l) for l in n_cell])} " +
          f"{' '.join(activation)}")
    print('###########################################################')
    print(' ')

    f_performance = open(DRIVE_DIR + skopt_param_file, 'a')
    f_performance.write(f"{loss} {n_conv} " +
                        f"{' '.join([str(l) for l in n_filters])} " +
                        f"{' '.join([str(l) for l in s_kernel])} " +
                        f"{' '.join([str(l) for l in stride])} " +
                        f"{' '.join([str(l) for l in dilation])} " +
                        f"{n_dense} " +
                        f"{' '.join([str(l) for l in n_cell])} " +
                        f"{' '.join(activation)}\n")
    f_performance.close()

    # k_back.clear_session()
    losses = np.append(losses, loss)
    np.save(DRIVE_DIR + 'losses_' + job_name + '.npy', losses)

    return loss


checkpoint_saver = CheckpointSaver(DRIVE_DIR + skopt_checkpoint_file, compress=9)

if load_flag:
    res = load(DRIVE_DIR + skopt_checkpoint_file)
    x0 = res.x_iters
    y0 = res.func_vals
else:
    # x0 = default_parameters
    x0 = None
    y0 = None

gp_result = gp_minimize(func=fitness,
                        dimensions=dimensions,
                        n_calls=10000,
                        noise=0.01,
                        n_jobs=-1,
                        kappa=5,
                        callback=[checkpoint_saver],
                        x0=x0,
                        y0=y0)
