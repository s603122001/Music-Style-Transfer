import numpy as np
import pickle

from project.utils import add_beat, padding


def generator(batch_size, subdivision, timesteps, step, dataset,
              phase='train', percentage_train=0.8,
              constraint=False
             ):

    if("jazz" in dataset):
        Y, X_metadatas = pickle.load(open(dataset, 'rb'))
    else:
        Y, X_metadatas, index2notes, note2indexs, metadatas = pickle.load(open(dataset, 'rb'))

    # Set chorale_indices
    if phase == 'train':
        chorale_indices = np.arange(int(len(Y) * percentage_train))
    if phase == 'test':
        chorale_indices = np.arange(int(len(Y) * percentage_train), len(Y))
    if phase == 'all':
        chorale_indices = np.arange(int(len(Y)))

    if("jazz" in dataset):
        for a in range(len(Y)):
            if (a in chorale_indices):
                new_y = np.zeros((Y[a].shape[:2]))
                for timeindex, beat in enumerate(Y[a]):
                    new_y[timeindex, np.nonzero(beat)[0]] = 1
                new_y = add_beat(new_y, subdivision)
                Y[a] = padding(new_y, timesteps, step)
            else:
                Y[a] = 0
    else:
        for a in range(len(Y)):
            if(a in chorale_indices):
                new_y = np.zeros(Y[a].shape[:2])
                conc_y = np.sum(Y[a], axis=2)
                for timeindex, beat in enumerate(conc_y):
                    new_y[timeindex][np.nonzero(beat)[0]] = 1
                new_y = add_beat(new_y,subdivision)
                Y[a] = padding(new_y, timesteps, step)
            else:
                Y[a] = 0

    central_features = []
    right_features = []
    left_features = []
    labels = []
    batch = 0
    midi_index = 0
    chorale_index = 0
    time_index = 0

    augmentation = 3
    non_zero = []
    non_zero_counter = 0
    while True:
        if(midi_index == 0 and augmentation == 3):
            chorale_index = np.random.choice(chorale_indices)
            chorale = np.array(Y[chorale_index])
            chorale_length = len(chorale)
            if(constraint == True):
                chorale_length = chorale_length - 2 * (timesteps * step)
                time_index = (timesteps * step) + np.random.randint(1, chorale_length // 4) * 4
            else:
                time_index = np.random.randint(timesteps * step, chorale_length - timesteps * step)
            non_zero = np.nonzero(Y[chorale_index][time_index, :-1])[0]
            if(len(non_zero) == 0):
                augmentation = 3
            else:
                augmentation = 0

        if(augmentation == 0 or augmentation == 2):
            midi_index = non_zero[non_zero_counter]
            non_zero_counter += 1

        central_feature = np.reshape(np.array(Y[chorale_index][time_index, :88]), (88, 1))
        central_feature[midi_index:] = 0.5

        left_feature = Y[chorale_index][time_index - (timesteps * step):time_index, :]
        right_feature = Y[chorale_index][(time_index + 1):(time_index + 1) + timesteps * step, :]

        label = np.zeros((2))
        label[int(Y[chorale_index][time_index, midi_index])] = 1

        central_features.append(central_feature)
        left_features.append(left_feature)
        right_features.append(right_feature)
        labels.append(label)

        batch += 1
        midi_index = (midi_index + 1) % 88

        if(augmentation == 1 and midi_index == 0):
            augmentation = 2
        if(augmentation == 0 or augmentation == 2):
            if(non_zero_counter == len(non_zero) and augmentation == 0):
                augmentation = 1
                midi_index = 0
                non_zero_counter = 0
            elif(non_zero_counter == len(non_zero) and augmentation == 2):
                augmentation = 3
                midi_index = 0
                non_zero_counter = 0

        # if there is a full batch
        if(batch == batch_size):
            next_element = (
                np.array(left_features, dtype=np.float32),
                np.array(central_features, dtype=np.float32),
                np.array(right_features, dtype=np.float32),

                np.array(labels, dtype=np.float32))

            yield next_element

            batch = 0

            central_features = []
            right_features = []
            left_features = []
            labels = []


def train(model,
          dataset_path,
          subdivision,
          epoch=80,
          steps=6000,
          timesteps=32,
          step=4,
          batch_size=88*3):
    generator_train = (({'left_features': left_features,
                         'central_features': central_features,
                         'right_features': right_features
                         },
                        {'prediction': labels})
                       for(left_features,
                           central_features,
                           right_features,
                           labels) in generator(batch_size, subdivision=subdivision,
                                                timesteps=timesteps, step=step,
                                                dataset=dataset_path, constraint=False
                                                ))

    generator_val = (({'left_features': left_features,
                       'central_features': central_features,
                       'right_features': right_features
                       },
                      {'prediction': labels})
                     for(left_features,
                         central_features,
                         right_features,
                         labels) in generator(batch_size, subdivision=subdivision,
                                              timesteps=timesteps, step=step, phase='test',
                                              dataset=dataset_path, constraint=False
                                              ))

    model.fit_generator(generator_train, samples_per_epoch=steps,
                        epochs=epoch, verbose=1, validation_data=generator_val,
                        validation_steps=200,
                        use_multiprocessing=False,
                        max_queue_size=100,
                        workers=1)


    return model