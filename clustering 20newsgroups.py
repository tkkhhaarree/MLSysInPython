import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk.stem
from sklearn.cluster import KMeans
import scipy as sp

english_stemmer = nltk.stem.SnowballStemmer("english")
all_sata = sklearn.datasets.fetch_20newsgroups(subset='all')
groups = ['comp.graphics', 'comp.os.ms-windows.misc',
          'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
          'comp.windows.x', 'sci.space']
train_data = sklearn.datasets.fetch_20newsgroups(subset='train',
                                                 categories=groups)
test_data = sklearn.datasets.fetch_20newsgroups(subset='test',
                                                categories=groups)


# trying Tfid vectorizer with stemming:
class StemmedTfidVectorizer(
    TfidfVectorizer):  # creating a new Term freq. inverse document Vectorizer class which extends original.
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in
                            analyzer(doc))  # automatically stem words when sentences are passed in class.


vectorizer = StemmedTfidVectorizer(min_df=10, max_df=0.5, stop_words='english', decode_error='ignore')
vectorized = vectorizer.fit_transform(train_data.data)
num_samples, num_features = vectorized.shape

# clustering the data.
num_clusters = 50
km = KMeans(n_clusters=num_clusters, init="random", n_init=1, verbose=1, random_state=3)
km.fit(vectorized)


new_post = "Disk drive problems. Hi, I have a problem with my hard disk.After 1 year it is working only sporadically now.I tried to format it, but now it doesn't boot any more.Any ideas? Thanks."
new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]  # best cluster for post to reside.

# finding distance between all posts within the best cluster and the new_post.
similar_indices = (km.labels_==new_post_label).nonzero()[0]
similar = []
for i in similar_indices:
    dist = sp.linalg.norm((new_post_vec -vectorized[i]).toarray())
    similar.append((dist, train_data.data[i]))
similar = sorted(similar)
print(similar[0]) # printing a post from the best cluster which is closest to new_post.
