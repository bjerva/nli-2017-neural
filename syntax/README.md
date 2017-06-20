# MaxLSTMClassifier

This system runs a biLSTM over each sentence in an essay, using the last state
in each direction (through a fully connected hidden layer) to predict the
language. The LSTM input consists of (embedded) POS tags concatenated with
token embeddings, which may (should) be initialized from a GloVe file.

## Data preparation

The essays need to be preprocessed with Stanford CoreNLP (tested with the
2017-06-09 release). Like this, for the dev set:

    ls /data1/corpora/nli-shared-task-2017/data/essays/dev/original/*.txt \
        >nli-dev.txt

    time java -cp "*" -Xmx4g edu.stanford.nlp.pipeline.StanfordCoreNLP \
        -annotators tokenize,ssplit,pos,lemma,ner,parse \
        -filelist nli-dev.txt \
        -outputDirectory \
            /data1/corpora/nli-shared-task-2017/data/essays/dev/parsed \
        -replaceExtension

Note that the XML files take a long time to load, so `train.py` will cache the
corpus in `train.pickle` and `dev.pickle` (currently hard-coded, this will
cause problems if you have multiple corpora).

## Training

    python3 train.py \
        --data-path nli-shared-task-2017 \
        --model models/runA \
        --gpu 0 \
        --batch-size 8 \
        --embeddings glove.840B.300d.txt \
        --embeddings-size 300 \
        --pos-embeddings-size 64 \
        --lstm-size 512 \
        | tee models/runA.log

