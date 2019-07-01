from load_dataset import *
import tensorflow as tf

class TrainModel:

    def __init__(self, epochs, note_tokenizer, sampled_200_midi, frame_per_second,
                 batch_nnet_size, batch_song, optimizer, loss_fn,
                 total_songs, model, seq_len):
        self.epochs = epochs
        self.note_tokenizer = note_tokenizer
        self.sampled_200_midi = sampled_200_midi
        self.frame_per_second = frame_per_second
        self.batch_nnet_size = batch_nnet_size
        self.batch_song = batch_song
        self.optimizer = optimizer
        #self.checkpoint = checkpoint
        self.loss_fn = loss_fn
        #self.checkpoint_prefix = checkpoint_prefix
        self.total_songs = total_songs
        self.model = model
        self.seq_len = seq_len

    def train(self):
        print("Training started...")
        for epoch in tqdm_notebook(range(self.epochs), desc='epochs'):
            # for each epochs, we shufle the list of all the datasets
            shuffle(self.sampled_200_midi)
            loss_total = 0
            steps = 0
            steps_nnet = 0

            # We will iterate all songs in the dataset with the step of batch_size
            for i in tqdm_notebook(range(0, self.total_songs, self.batch_song), desc='MUSIC'):

                steps += 1
                inputs_nnet_large, outputs_nnet_large = generate_batch_song(
                    self.sampled_200_midi, self.batch_song, start_index=i, fs=self.frame_per_second,
                    seq_len=self.seq_len, use_tqdm=False)  # We use the function that have been defined here
                outputs_nnet_large = np.array(self.note_tokenizer.transform(outputs_nnet_large), dtype=np.int32)

                index_shuffled = np.arange(start=0, stop=len(inputs_nnet_large))
                np.random.shuffle(index_shuffled)

                for nnet_steps in tqdm_notebook(range(0, len(index_shuffled), self.batch_nnet_size)):
                    steps_nnet += 1
                    current_index = index_shuffled[nnet_steps:nnet_steps + self.batch_nnet_size]
                    inputs_nnet, outputs_nnet = inputs_nnet_large[current_index], outputs_nnet_large[current_index]

                    # To make sure no exception thrown by tensorflow on autograph
                    if len(inputs_nnet) // self.batch_nnet_size != 1:
                        break
                    loss = self.train_step(inputs_nnet, outputs_nnet)
                    loss_total += tf.math.reduce_sum(loss)
                    if steps_nnet % 20 == 0:
                        print("epoch {} | Steps {} | total loss : {}".format(epoch + 1, steps_nnet, loss_total))


    @tf.function
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            prediction = self.model(inputs)
            loss = self.loss_fn(targets, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss