# TP 1 MACHINE LEARNING 



###Veuillez consulter le nootebook pour voir toutes les etapes du projet
````python
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler


````

### Importation des données
````python
positif = pd.read_pickle(r'data/imdb_raw_pos.pickle')
negatif = pd.read_pickle(r'data/imdb_raw_neg.pickle')
print('positif comment',len(positif))
print('negatif comment', len(negatif))

````