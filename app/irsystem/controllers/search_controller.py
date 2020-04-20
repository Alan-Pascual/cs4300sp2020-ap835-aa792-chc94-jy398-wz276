from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

# Libraries for Search
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import re
import requests

project_name = "Culture for Quarantined Gamers: Anime Recommendations Based on Game Preferences"
net_id = "Amrit Amar (aa792),  Carina Cheng (chc94), Alan Pascual (ap835), Jeffrey Yao (jy398), Wenjia Zhang (wz276)"
TAG_RE = re.compile(r'<[^>]+>')

def createModel(file):
	with open(file) as f:
    		raw_docs = json.loads(f.readlines()[0])
		
	documents = []
	for anime in raw_docs["shows"]:
		reviews = ""
		for review in anime['reviews']:
			reviews += review['content']
		documents.append( (anime['title'], anime['description'], reviews, anime['image_url'], anime['promo_url']) )

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

def getGames():
	url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
	r = requests.get(url)
	data = r.json()
	appList = data["applist"]["apps"]

	tupleList = np.array([(0, "")] * len(appList), dtype='object')
	for i, app in enumerate(appList):
		tupleList[i] = tuple([int(app['appid']), str(app['name'])])

	return tupleList

def getSimilarNames(gamesList, query : str):
    similarNames = []
    for (appId, name) in gamesList:
        if query.lower() == name.lower():
            similarNames += [(appId, name)]
            
    if len(similarNames) == 0:
        for (appId, name) in gamesList:
            if query.lower() in name.lower():
                similarNames += [(appId, name)]
                
    return np.array(similarNames)

def remove_tags(text):
    return TAG_RE.sub('', text)

def getGamesDesciription(id):
    url = "https://store.steampowered.com/api/appdetails?appids=" + str(id)
    r = requests.get(url)
    data = r.json()
    appList = data[str(id)]['data']['detailed_description']
    
    return remove_tags(appList)

def getAnimeList(game, gameList, id=False):
    if id:
        gameID = game
    else:
        gameID = getSimilarNames(gameList, game)
    if len(gameID) == 0:
        return "XD?"
    else:
        gameID = gameID[0][0]
    desc = getGamesDesciription(gameID)
    
    animeList = []
    
    #Tokenize the Description
    desc = desc.lower().split()
    
    for word in desc:
        word_list = closest_project_to_word(word.lower(), 5)
        if word_list != "Not in vocab.":
            for anime in word_list: #for each anime in list of anime
                found = False
                for i, animeClosest in zip(range(len(animeList)), animeList): 
                    if animeClosest[0] == anime[0]: #found anime
                        animeList[i][1] += anime[1]
                        found = True
                if not found:
                    animeList.append([anime[0], anime[1]])
                    
    final_list = sorted(animeList, key = lambda x: float(x[1]), reverse = True)
    final_list = [x[0] for x in final_list]
            
    return final_list[:5]

def getAnimeInfo(AnimeName):
    record = []
    for anime in documents:
        if AnimeName == anime[0]:
            record = [anime[0], anime[1], anime[3].split('?')[0], anime[4].split('?')[0]]
            break
    return record

gameList = getGames()

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		descriptions = []
		pictures = []
		videos = []
		output_message = ''
	else:
		output_message = "Your search: " + query + " (this will be replaced with anime)"
		
		closestAnime = getAnimeList(query, gameList)
		if closestAnime == "XD?":
			data = ["Something went wrong!"]
			descriptions = ["Something went wrong!"]
			pictures = ["Something went wrong!"]
			videos = ["Something went wrong!"]
		else:
			info_anime = []
			for anime in closestAnime:
				info_anime.append(getAnimeInfo(anime))

			data = [x[0] for x in info_anime]
			descriptions = [x[1] for x in info_anime]
			pictures = [x[2] for x in info_anime]
			videos = [x[3] for x in info_anime]
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data, descriptions=descriptions, pictures=pictures, videos=videos)



