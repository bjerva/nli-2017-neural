from chainer import Chain
import chainer.functions as F
import chainer.links as L

class MaxLSTMClassifier(Chain):
    def __init__(self, alphabet_size, pos_alphabet_size,
                 embedding_size, pos_embedding_size,
                 state_size, n_classes,
                 embeddings=None, dropout=0.0):
        super().__init__(
                embeddings=L.EmbedID(
                    alphabet_size, embedding_size, ignore_label=-1,
                    initialW=embeddings),
                pos_embeddings=L.EmbedID(
                    pos_alphabet_size, pos_embedding_size, ignore_label=-1),
                lstm=L.NStepBiLSTM(2, embedding_size+pos_embedding_size,
                                   state_size, dropout),
                hidden=L.Linear(state_size*2, 1024),
                output=L.Linear(1024, n_classes))
        self.state_size = state_size

    def __call__(self, sequence_lists, pos_sequence_lists):
        embedded_sequences = [
                self.embeddings(sequence)
                for sequences in sequence_lists
                for sequence in sequences]

        pos_embedded_sequences = [
                self.pos_embeddings(sequence)
                for sequences in pos_sequence_lists
                for sequence in sequences]

        embedded_sequences = [
                F.concat(embedded, axis=1)
                for embedded in zip(embedded_sequences, pos_embedded_sequences)]

        _,_,encoded = self.lstm(None, None, embedded_sequences)
        # TODO: this could probably be done more efficiently with separate
        # LSTMs
        state_size = self.state_size
        encoded = F.stack([
            F.concat((sequence[-1,:state_size], sequence[0,state_size:]),
                     axis=0)
            for sequence in encoded], axis=0)

        i = 0
        pooled = []
        for sequences in sequence_lists:
            pooled.append(F.max(encoded[i:i+len(sequences)], axis=0))
            i += len(sequences)

        pooled = F.stack(pooled, axis=0)
        hidden = F.dropout(F.relu(self.hidden(pooled)), 0.5)
        return self.output(hidden)

