import tensorflow as tf
import pickle
import pretty_midi
import numpy as np
from tqdm import tqdm_notebook
from musthe import *
import random

def number_to_note(number):
    octave = number // len(all_notes)
    note = all_notes[number % len(all_notes)]
    return note + str(octave - 1)


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def generate_from_one_note(note_tokenizer, new_notes='35'):
    generate = [note_tokenizer.notes_to_index['e'] for _ in range(49)]
    generate += [note_tokenizer.notes_to_index[new_notes]]
    return generate


def generate_notes(generate, model, unique_notes, max_generated=1000, seq_len=50):
  for i in tqdm_notebook(range(max_generated), desc='genrt'):
    test_input = np.array([generate])[:, i:i+seq_len]
    predicted_note = model.predict(test_input)
    random_note_pred = np.random.choice(unique_notes+1, 1, replace=False, p=predicted_note[0])
    generate.append(random_note_pred[0])
  return generate


def write_midi_file_from_generated(generate, midi_file_name="result.mid", start_index=49, fs=8, max_generated=1000):
    note_string = [note_tokenizer.index_to_notes[ind_note] for ind_note in generate]
    array_piano_roll = np.zeros((128, max_generated + 1), dtype=np.int16)
    for index, note in enumerate(note_string[start_index:]):
        if note == 'e':
            pass
        else:
            splitted_note = note.split(',')
            for j in splitted_note:
                array_piano_roll[int(j), index] = 1
    generate_to_midi = piano_roll_to_pretty_midi(array_piano_roll, fs=fs)
    print("Tempo {}".format(generate_to_midi.estimate_tempo()))
    for note in generate_to_midi.instruments[0].notes:
        note.velocity = 100
    generate_to_midi.write(midi_file_name)


def filter_notes_by_scale(note, scale_key):
    scale = Scale(Note(note), scale_key)
    index_to_notes_dict = note_tokenizer.index_to_notes
    num_indices = len(index_to_notes_dict) + 1
    for i in reversed(range(1, num_indices)):
        note_number = index_to_notes_dict[i]
        if note_number == 'e' or (',' not in note_number and 0 <= int(note_number) <= 127):
            if note_number != 'e':
                if Note(number_to_note(int(note_number))) not in scale:
                    index_to_notes_dict[i] = str(int(index_to_notes_dict[i]) + random.choice([-1, 1]))


all_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
octaves = list(range(11))

model = tf.keras.models.load_model('model_ep4.h5')
note_tokenizer = pickle.load(open("tokenizer.p", "rb"))

filter_notes_by_scale('D', scale_key='major')

max_generate = 500
unique_notes = note_tokenizer.unique_word
seq_len = 50

for i in range(3):
    generate = generate_from_one_note(note_tokenizer, '55')
    generate = generate_notes(generate, model, unique_notes, max_generate, seq_len)
    write_midi_file_from_generated(generate, "one_note" + str(i) + ".mid", start_index=seq_len - 1, fs=15, max_generated=max_generate)
