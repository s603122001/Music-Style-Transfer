import numpy as np
import argparse

from project.midi_handler import midi2score, score2midi
from project.utils import padding, load_model, save_model, add_beat
from project.test import style_transfer
from project.model import lstm_wavenet
from project.train import train

def main():
    # Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--phase',
                        help='phase: training or testing (default: %(default)s',
                        type=str, default='testing')
    # arguments for testing
    parser.add_argument('-d', '--dataset_path',
                        help='path to data set (default: %(default)s',
                        type=str, default='bach_dataset.pickle')

    parser.add_argument('-e', '--epoch',
                        help='number of epoch(default: %(default)s',
                        type=int, default=80)
    parser.add_argument('-n', '--steps',
                        help='number of step per epoch(default: %(default)s',
                        type=int, default=6000)
    parser.add_argument('-b', '--batch_size_train',
                        help='batch size(default: %(default)s',
                        type=int, default=88*3)
    parser.add_argument('-o', '--output_model_name',
                        help='name of the output model(default: %(default)s',
                        type=str, default="out")
    # arguments for testing
    parser.add_argument('-m', '--model_path',
                        help='path to existing model (default: %(default)s',
                        type=str, default='bach')
    parser.add_argument('-i', '--input_file',
                        help='path to input file (default: %(default)s',
                        type=str, default="LiveAndLetDie_all.mid")
    parser.add_argument('-ii', '--input_file_melody',
                        help='path to input melody file (default: %(default)s',
                        type=str, default="LiveAndLetDie_main.mid")
    parser.add_argument('-s', '--subdivision',
                        help='subdivision within one beat (default: %(default)s',
                        type=int, default=4)

    args = parser.parse_args()
    print(args)

    if(args.phase == "training"):
        #set arguments

        timesteps = 32
        step = 4
        subdivision = args.subdivision
        batch_size = args.batch_size_train
        dataset_path = args.dataset_path

        #create model

        model = lstm_wavenet(num_features_lr=91, timesteps=timesteps,
                             step=step, num_units_lstm=[150, 150, 150, 150],
                             num_dense=150,
                             conv_layers=5,
                             skip_layers=2)

        model.compile(optimizer="adam", loss={'prediction': 'binary_crossentropy'}, metrics=['accuracy'])

        #train

        model = train(model,
                      dataset_path,
                      subdivision,
                      epoch=args.epoch,
                      steps=args.steps,
                      timesteps=timesteps,
                      step=step,
                      batch_size=batch_size)
        #save model

        save_model(model, args.output_model_name)

    else:
        #load input file

        subdivision = args.subdivision
        path = args.input_file
        path_melody = args.input_file_melody
        score = midi2score(path, subdivision)

        if(path_melody == "none"):
            score_melody = np.zeros(score.shape)
        else:
            score_melody = midi2score(path_melody, subdivision)

        score = add_beat(score, subdivision)
        score_melody = add_beat(score_melody, subdivision)

        score = np.array(score[0:640])
        score_melody = np.array(score_melody[0:640])

        extended_score = padding(score, 32, 4)

        #load model

        model = load_model(model_name=args.model_path)

        #generation

        result = style_transfer(extended_score, score_melody, model, iter_num=25)

        #save result

        score2midi("test.mid", result, subdivision, 120, melody_constraint=True, melody=score_melody)
        print("saved")

if __name__ == "__main__":
    main()