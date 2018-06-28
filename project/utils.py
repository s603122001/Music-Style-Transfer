import numpy as np
from keras.models import model_from_json, model_from_yaml


def padding(x, timesteps, step):
    extended_chorale = np.array(x)
    padding_dimensions = (timesteps * step,) + extended_chorale.shape[1:]

    padding_start = np.zeros(padding_dimensions)
    padding_end = np.zeros(padding_dimensions)
    for a, b in enumerate(padding_start):
        padding_start[a][-3] = 1
        padding_end[a][-2] = 1

    extended_chorale = np.concatenate((padding_start,
                                       extended_chorale,
                                       padding_end),
                                      axis=0)
    return extended_chorale


def padding_ade(x, timesteps, subdivision, step, gap):
    # extended_chorale = np.concatenate((-1*np.ones((x.shape[0],88)),x), axis = 1)
    extended_chorale = np.concatenate((x, np.zeros((x.shape[0], 1))), axis=1)

    for i in range(len(extended_chorale)):
        extended_chorale[i][-1] = i % subdivision + 1

    padding_dimensions = (timesteps * step + gap,) + extended_chorale.shape[1:]

    padding_start = np.zeros(padding_dimensions)
    padding_end = np.zeros(padding_dimensions)
    for a, b in enumerate(padding_start):
        padding_start[a][-3] = 1
        padding_end[a][-2] = 1

    extended_chorale = np.concatenate((padding_start,
                                       extended_chorale,
                                       padding_end),
                                      axis=0)

    return extended_chorale


def add_beat(score, subdivision):
    b = np.zeros((score.shape[0], 1))
    score = np.concatenate([score, b], axis=1)

    for time in range(0, len(score)):
        score[time][-1] = time % subdivision + 1

    return score

def load_model(model_name):
    """
    """
    ext = '.yaml'
    model = model_from_yaml(open(model_name + ext).read())
    model.load_weights(model_name + '_weights.h5')

    print("model " + model_name + " loaded")
    return model


def save_model(model, model_name, overwrite=False):
    # SAVE MODEL

    string = model.to_yaml()
    ext = '.yaml'

    open(model_name + ext, 'w').write(string)
    model.save_weights(model_name + '_weights.h5', overwrite=overwrite)
    print("model " + model_name + " saved")