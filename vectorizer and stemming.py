import os, sys, nltk.stem
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# words which are to be ignored during analysis.
stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again',
             'against', 'all', 'almost', 'alone', 'along', 'already', 'also',
             'although', 'always', 'am', 'among', 'amongst', 'amoungst']
posts = [open(os.path.join("toy", f)).read() for f in os.listdir("toy")]  # copy content of text files in elements of list.
vectorizer = CountVectorizer(min_df=1, stop_words=stopwords)
X_train = vectorizer.fit_transform(
    posts)  # creating vectors from post data according to frequency of each letter in each post.
print("vectorized data: ", X_train)
num_samples, num_features = X_train.shape  # no. of distinct words, no. of text files.
new_post = "imaging databases" # query string

new_post_vec = vectorizer.transform([new_post])
# print no. of occurrences of vector [x, y] where x is index of element taken in array as input by transform func., y is feature index.
# for example, output (0, 4)  1 means that 4th word from the 0th file comes once in the new_post string.
print("vector for new_post string: \n",new_post_vec)


# compute the distance between two normalised vectors.
def dist_raw(v1, v2):
    v1_normalised = v1 / sp.linalg.norm(v1.toarray())  # vector divided by its magnitude.
    v2_normalised = v2 / sp.linalg.norm(v2.toarray())
    delta = v1_normalised - v2_normalised
    return sp.linalg.norm(delta.toarray())


best_doc = None
best_dist = sys.maxsize
best_i = None
for i in range(num_samples):
    if posts[i] == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_raw(post_vec, new_post_vec)
    if d < best_dist:
        best_dist = d
        best_i = i
print("for count vectorizer, best post is %i.txt with dist=%.2f" % (best_i+1, best_dist))

english_stemmer = nltk.stem.SnowballStemmer("english")


class StemmedCountVectorizer(CountVectorizer):  # creating a new StemmedCountVectorizer class which extends CountVectorizer.
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in
                            analyzer(doc))  # automatically stem words when sentences are passed in class.


vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')
X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape
new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])

best_doc = None
best_dist = sys.maxsize
best_i = None
for i in range(num_samples):
    if posts[i] == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_raw(post_vec, new_post_vec)
    if d < best_dist:
        best_dist = d
        best_i = i
print("for stemmed count vectorizer, best post is %i.txt with dist=%.2f" % (best_i+1, best_dist))


# trying Tfid vectorizer with stemming:
class TfidStemmedCountVectorizer(TfidfVectorizer):  # creating a new Term freq. inverse document Vectorizer class which extends original.
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in
                            analyzer(doc))  # automatically stem words when sentences are passed in class.


vectorizer = TfidStemmedCountVectorizer(min_df=1, stop_words='english')
X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape
new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])

best_doc = None
best_dist = sys.maxsize
best_i = None
for i in range(num_samples):
    if posts[i] == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_raw(post_vec, new_post_vec)
    if d < best_dist:
        best_dist = d
        best_i = i
print("for stemmed tfid vectorizer, best post is %i.txt with dist=%.2f" % (best_i+1, best_dist))
