from load_dataset import *
import tensorflow as tf
from train_model import TrainModel
import pickle
from note_tokenizer import NoteTokenizer

list_all_midi = get_list_midi()
sampled_200_midi = list_all_midi[0:200]

seq_len = 50
EPOCHS = 5
BATCH_SONG = 16
BATCH_NNET_SIZE = 96
TOTAL_SONGS = len(sampled_200_midi)
FRAME_PER_SECOND = 5

note_tokenizer = NoteTokenizer()

print("Loading and converting MIDI files...")
for i in tqdm_notebook(range(len(sampled_200_midi))):
    dict_time_notes = generate_dict_time_notes(sampled_200_midi, batch_song=1, start_index=i, use_tqdm=False, fs=5)
    full_notes = process_notes_in_song(dict_time_notes)
    for note in full_notes:
        note_tokenizer.partial_fit(list(note.values()))

note_tokenizer.add_new_note('e')
unique_notes = note_tokenizer.unique_word  # output = unique_notes + 1 because of the pause note


def create_model(seq_len, unique_notes, dropout=0.3, output_emb=100, rnn_unit=256, dense_unit=128):
    inputs = tf.keras.layers.Input(shape=(seq_len,))
    embedding = tf.keras.layers.Embedding(input_dim=unique_notes+1, output_dim=output_emb, input_length=seq_len)(inputs)
    forward_pass = tf.keras.layers.LSTM(rnn_unit, return_sequences=True)(embedding)
    forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = tf.keras.layers.LSTM(rnn_unit)(forward_pass)
    forward_pass = tf.keras.layers.Dense(dense_unit)(forward_pass)
    forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    outputs = tf.keras.layers.Dense(unique_notes+1, activation="softmax")(forward_pass)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='generate_scores_rnn')
    return model


model = create_model(seq_len, unique_notes)
model.summary()

optimizer = tf.keras.optimizers.RMSprop()
loss_fn = tf.keras.losses.categorical_crossentropy

train_class = TrainModel(EPOCHS, note_tokenizer, sampled_200_midi, FRAME_PER_SECOND,
                  BATCH_NNET_SIZE, BATCH_SONG, optimizer, loss_fn, TOTAL_SONGS, model, seq_len)

train_class.train()
model.save('model_ep4.h5')
pickle.dump(note_tokenizer, open("tokenizer.p", "wb"))

