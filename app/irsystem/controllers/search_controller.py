from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

# Libraries for Search
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

project_name = "Culture for Quarantined Gamers: Anime Recommendations Based on Game Preferences"
net_id = "Amrit Amar (aa792),  Carina Cheng (chc94), Alan Pascual (ap835), Jeffrey Yao (jy398), Wenjia Zhang (wz276)"

def createModel(file):
	with open(file) as f:
    		raw_docs = json.loads(f.readlines()[0])
		
	documents = []
	for anime in raw_docs["shows"]:
		reviews = ""
		for review in anime['reviews']:
			reviews += review['content']
		documents.append( (anime['title'], anime['description'], reviews) )

	np.random.shuffle(documents)
	return documents
	
documents = createModel('.'+os.path.sep+'anime_data1.json')
vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .9, min_df = 2)
my_matrix = vectorizer.fit_transform([x[2] for x in documents]).transpose()

words_compressed, _, docs_compressed = svds(my_matrix, k=100) 
docs_compressed = docs_compressed.transpose()
	
word_to_index = vectorizer.vocabulary_
index_to_word = {i:t for t,i in word_to_index.items()}
	
words_compressed = normalize(words_compressed, axis = 1)
	
def closest_words(word_in, k = 10):
	if word_in not in word_to_index: return [("Not in vocab.", 0)]
	sims = words_compressed.dot(words_compressed[word_to_index[word_in],:])
	asort = np.argsort(-sims)[:k+1]
	return [(index_to_word[i],sims[i]/sims[asort[0]]) for i in asort[1:]]

docs_compressed = normalize(docs_compressed, axis = 1)
def closest_project_to_word(word_in, k = 5):
	if word_in not in word_to_index: return [("Not in vocab.", 0)]
	sims = docs_compressed.dot(words_compressed[word_to_index[word_in],:])
	asort = np.argsort(-sims)[:k+1]
	return [(documents[i][0], sims[i]/sims[asort[0]]) for i in asort[1:]]

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ''
	else:
		output_message = "Your search: " + query + " (this will be replaced with anime)"
		closestAnime = closest_project_to_word(query, 5)
		
		data = [x[0] for x in closestAnime]
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)



