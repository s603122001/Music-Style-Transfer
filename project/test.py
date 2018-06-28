import numpy as np
import tqdm


def time_index_chosing(_range,
                       interval,
                       random=True,
                       time_list=None):
    interval += 1
    if (random):
        time_index_base = np.random.randint(_range)
    else:
        time_index_base = time_list[np.random.randint(len(time_list))]

    c = time_index_base + np.arange(-interval * 200, interval * 200, interval)

    return c[np.where(np.logical_and(c >= 0, c < _range))]


def midi_index_chosing(_len):
    return np.random.randint(88, size=_len)


def generation_wavenet(model,
                       score,
                       time_indexes,
                       midi_indexes,
                       timesteps=32,
                       step=4
                       ):
    time_indexes = np.array(time_indexes) + timesteps * step

    left_features = np.array(score[[np.arange(t - (timesteps * step), t) for t in time_indexes], :])
    right_features = np.array(score[[np.arange(t + 1, t + 1 + (timesteps * step)) for t in time_indexes], :])
    central_features = np.reshape(np.array(score[time_indexes, :88]), (len(time_indexes), 88, 1))

    for a, b in enumerate(midi_indexes):
        central_features[a, b:] = 0.5

    p = model.predict([left_features, central_features, right_features])

    return p


def style_transfer(extended_score, score_melody,
                   model,
                   iter_num=25,
                   timesteps=32,
                   step=4,
                   threshold=0.5
                   ):

    fixed_rhythm_score = score_melody
    original_len = len(score_melody)
    new_extended_score = np.array(extended_score)
    counter = 0
    alpha_initial = 0.6
    alpha = alpha_initial
    alpha_min = 0
    annealing_fraction = 0.6
    update_count = 0

    for i in tqdm.tqdm(range(iter_num)):
        time_list = np.arange(original_len)
        print("alpha = ", alpha)
        while (time_list.size > 0):
            if(alpha != -1):
                alpha = max(0, alpha_initial - update_count * (alpha_initial - alpha_min) / (
                    iter_num * original_len * annealing_fraction))
            if(alpha == 0):
                extended_score = new_extended_score
                alpha = -1
            elif(counter / original_len > alpha and alpha != -1):
                counter = 0
                extended_score = np.array(new_extended_score)

            time_indexes = time_index_chosing(original_len, timesteps * step, random=False, time_list=time_list)
            l = len(time_indexes)
            sorter = np.argsort(time_list)
            d = sorter[np.searchsorted(time_list, time_indexes, sorter=sorter)]
            time_list = np.delete(time_list, d, 0)
            counter += l

            update_count += l

            if(alpha != -1):
                midi_indexes = np.arange(88).tolist() * len(time_indexes)
                time_indexes_repeat = np.repeat(time_indexes, 88)
                p = generation_wavenet(model, extended_score, time_indexes_repeat, midi_indexes,
                                       timesteps=timesteps, step=step)
                for i, t in enumerate(time_indexes_repeat):
                    if(fixed_rhythm_score[t, midi_indexes[i]] == 0):
                        if(p[i][1] > threshold):
                            new_extended_score[t + timesteps * step, midi_indexes[i]] = 1
                        elif(p[i][0] > threshold):
                            new_extended_score[t + timesteps * step, midi_indexes[i]] = 0

            else:
                for midi_index in range(88):
                    midi_indexes = [midi_index] * l
                    p = generation_wavenet(model, extended_score, time_indexes, midi_indexes,
                                           timesteps=timesteps, step=step)
                    for i, t in enumerate(time_indexes):
                        if(fixed_rhythm_score[t, midi_indexes[i]] == 0):
                            if(p[i][1] > threshold):
                                new_extended_score[t + timesteps * step, midi_indexes[i]] = 1
                            elif(p[i][0] > threshold):
                                new_extended_score[t + timesteps * step, midi_indexes[i]] = 0


    return new_extended_score[timesteps*step:-timesteps*step,:88]