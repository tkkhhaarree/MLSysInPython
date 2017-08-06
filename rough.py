from gensim import corpora, models
import matplotlib.pyplot as plt
# ap.txt file contains many text documents, vocab.txt file is used to assign an id to each word appearing in ap.txt (line no. is id),
# ap.dat file contains data in format:
# n id_1:freq_1 id_2:freq_2...id_n:freq_n where line no. is doc. no., n is no. of words in document, and each word's id is mapped to its frequency.
corpus = corpora.BleiCorpus('./data/ap/ap.dat', './data/ap/vocab.txt')

model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word)
doc = corpus.docbyoffset(0)
topics = model[doc]
num_topics=len(topics)
plt.hist(num_topics)
plt.show()