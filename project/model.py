from keras.engine import Input, Model
from keras.layers import Dense, Reshape, Permute, add, TimeDistributed, LSTM, CuDNNLSTM, Dropout, Lambda, concatenate, Multiply
from keras.layers.core import Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D,Conv2D,Conv2DTranspose,MaxPooling2D,Cropping2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam


def lstm_wavenet(num_features_lr=91,
                     num_pitches=88,
                     num_units_lstm=[150, 150, 150, 150],
                     num_dense=90,
                     timesteps=32,
                     step=4,
                     conv_layers=5,
                     skip_layers=2,
                     using_cuda=True
                     ):

    SelectedLSTM = CuDNNLSTM if using_cuda else LSTM

    left_features = Input(shape=(timesteps * step, num_features_lr), name='left_features')
    left_features_reshape = Reshape((timesteps, num_features_lr * step))(left_features)
    right_features = Input(shape=(timesteps * step, num_features_lr), name='right_features')
    right_features_reshape = Reshape((timesteps, num_features_lr * step))(right_features)
    central_features = Input(shape=(88, 1), name='central_features')

    embedding_left = Dense(input_dim=num_features_lr * step,
                           units=num_dense, name='embedding_left')
    embedding_right = Dense(input_dim=num_features_lr * step,
                            units=num_dense, name='embedding_right')

    predictions_left = left_features_reshape
    predictions_right = right_features_reshape

    # input dropout
    predictions_left = Dropout(0.3)(predictions_left)
    predictions_right = Dropout(0.3)(predictions_right)
    # embedding
    predictions_left = TimeDistributed(embedding_left)(predictions_left)
    predictions_right = TimeDistributed(embedding_right)(predictions_right)
    # left recurrent networks
    return_sequences = True
    for k, stack_index in enumerate(range(len(num_units_lstm))):
        if k == len(num_units_lstm) - 1:
            return_sequences = False
        if k > 0:
            predictions_left_tmp = add([LeakyReLU(alpha=0.2)(predictions_left), predictions_left_old])
            predictions_right_tmp = add([LeakyReLU(alpha=0.2)(predictions_right), predictions_right_old])
        else:
            predictions_left_tmp = predictions_left
            predictions_right_tmp = predictions_right

        predictions_left_old = predictions_left
        predictions_left = predictions_left_tmp
        predictions_left = SelectedLSTM(num_units_lstm[stack_index],
                                        return_sequences=return_sequences,
                                        name='lstm_left_' + str(stack_index)
                                        )(predictions_left)

        predictions_right_old = predictions_right
        predictions_right = predictions_right_tmp
        predictions_right = SelectedLSTM(num_units_lstm[stack_index],
                                         return_sequences=return_sequences,
                                         name='lstm_right_' + str(stack_index)
                                         )(predictions_right)
        # LSTM Dropout
        predictions_left = Dropout(0.3)(predictions_left)
        predictions_right = Dropout(0.3)(predictions_right)

    # retain only last input for skip connections
    predictions_left_old = Lambda(lambda t: t[:, -1, :],
                                  output_shape=lambda input_shape: (input_shape[0], input_shape[-1])
                                  )(predictions_left_old)

    predictions_right_old = Lambda(lambda t: t[:, -1, :],
                                   output_shape=lambda input_shape: (input_shape[0], input_shape[-1])
                                   )(predictions_right_old)
    # concat or sum
    predictions_left = concatenate([LeakyReLU(alpha=0.2)(predictions_left), predictions_left_old])
    predictions_right = concatenate([LeakyReLU(alpha=0.2)(predictions_right), predictions_right_old])
    predictions_context = concatenate([predictions_left, predictions_right])
    predictions_context = LeakyReLU(alpha=0.2)(Dense(num_pitches)(predictions_context))
    predictions_context = Reshape((num_pitches, 1))(predictions_context)
    # wavnet part
    skip = central_features
    skips = []
    for i in range(conv_layers):
        conv_central_t = BatchNormalization()(Conv1D(64, 2, dilation_rate=2 ** (i), padding='causal')(skip))
        conv_central_s = BatchNormalization()(Conv1D(64, 2, dilation_rate=2 ** (i), padding='causal')(skip))
        if (i < skip_layers):
            conv_context_t = BatchNormalization()(
                Conv1D(64, 2, dilation_rate=2 ** (i), padding='causal')(predictions_context))
            conv_context_s = BatchNormalization()(
                Conv1D(64, 2, dilation_rate=2 ** (i), padding='causal')(predictions_context))
            conv_t = Activation('tanh')(add([conv_central_t, conv_context_t]))
            conv_s = Activation('sigmoid')(add([conv_central_s, conv_context_s]))
            conv = Multiply()([conv_t, conv_s])
            conv = BatchNormalization()(Conv1D(1, 1, padding='same')(conv))
            skip = add([conv, skip])
        else:
            conv_t = Activation('tanh')(conv_central_t)
            conv_s = Activation('sigmoid')(conv_central_s)
            conv = Multiply()([conv_t, conv_s])
            conv = BatchNormalization()(Conv1D(1, 1, padding='same')(conv))
            skip = add([conv, skip])
        skips.append(conv)

    out = LeakyReLU(alpha=0.2)(add(skips))
    out = LeakyReLU(alpha=0.2)(Conv1D(1, 1)(out))
    out = Flatten()(Conv1D(1, 1)(out))
    out = Dense(2, activation='softmax', name='prediction')(out)

    model = Model(inputs=[left_features, central_features, right_features],
                  outputs=out)
    model.compile(optimizer='adam',
                  loss={'prediction': 'binary_crossentropy'},
                  )

    return model