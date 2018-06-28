import mido
from mido import MetaMessage, Message, MidiFile, MidiTrack

import numpy as np


def midi2score(song,
               subdivision):
    mid = mido.MidiFile(song)

    tempo = 0
    sec_per_tick = 0
    length = mid.length

    time = 0

    # set initial score len
    for msg in mid:
        if (msg.is_meta):
            if (msg.type == 'set_tempo'):
                tempo = msg.tempo
        else:
            if (msg.type == "note_on"):
                bpm = mido.tempo2bpm(tempo)
                sec_per_tick = 60 / bpm / subdivision
                break

    score = np.zeros((int(length / sec_per_tick) + 1, 90))

    for msg in mid:
        time += msg.time
        pos = int(np.round(time / sec_per_tick))
        if (pos + 1 > score.shape[0]):
            score = np.append(score, np.zeros((pos - score.shape[0] + 1, 90)), axis=0)
        if (msg.is_meta):
            if (msg.type == 'set_tempo'):
                tempo = mido.tempo2bpm(msg.tempo)
                sec_per_tick = 60 / tempo / subdivision

        elif (msg.type == 'note_on'):
            if (msg.velocity == 0):
                p = msg.note - 21
                score[pos:, p] = 0
            else:
                p = msg.note - 21
                score[pos:, p] = 1

        elif (msg.type == 'note_off'):
            p = msg.note - 21
            score[pos:, p] = 0

    return score


def score2midi(name,
               score,
               subdivision,
               bpm,
               melody_constraint=False,
               melody=None):

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # meta messages
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    # note messages
    ticks_per_beat = 480
    ticks_per_division = int(ticks_per_beat // subdivision)
    pitch_table = np.zeros((88, 1))

    for index, tick in enumerate(score):
        # offsets handle
        current_notes = np.nonzero(pitch_table)[0]
        for _note in current_notes:
            if (tick[_note] == 0):
                track.append(Message('note_on', note=_note + 21, velocity=0, time=0))
                pitch_table[_note] = 0

        # onsets handle
        # melody constraint handle
        onsets = np.nonzero(tick[:])[0]
        for _note in onsets:
            if (melody_constraint == True):
                if (melody[index][_note] == 1):
                    if (pitch_table[_note] == 1):
                        track.append(Message('note_on', note=_note + 21, velocity=0, time=0))
                    if (pitch_table[_note] != 2):
                        track.append(Message('note_on', note=_note + 21, velocity=85, time=0))
                        pitch_table[_note] = 2
                elif (pitch_table[_note] == 0):
                    track.append(Message('note_on', note=_note + 21, velocity=80, time=0))
                    pitch_table[_note] = 1
            else:
                if (pitch_table[_note] == 0):
                    track.append(Message('note_on', note=_note + 21, velocity=80, time=0))
                    pitch_table[_note] = 1

        # time progress
        track.append(Message('note_on', note=0, velocity=0, time=ticks_per_division))

    mid.save(name)