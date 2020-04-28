from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

# Libraries for Search
import numpy as np
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import re
import requests
import pickle
import time

project_name = "Games2Anime: Anime Recommendations Based on Game Preferences"
net_id = "Amrit Amar (aa792),  Carina Cheng (chc94), Alan Pascual (ap835), Jeffrey Yao (jy398), Wenjia Zhang (wz276)"
TAG_RE = re.compile(r'<[^>]+>')
genre_dict = {1: 'Action', 2: 'Adventure', 3: 'Cars',4: 'Comedy',5: 'Dementia',6: 'Demons',7: 'Mystery',8: 'Drama', 9: 'Ecchi',10:'Fantasy',11:'Game',12:'Hentai',13:'Historical',14:'Horror',15:'Kids',16:'Magic',17:'Martial Arts',18:'Mecha',19:'Music',20:'Parody',21:'Samurai',22:'Romance',23:'School',24:'Sci-Fi',25:'Shoujo',26:'Shoujo Ai',27:'Shounen',28:'Shounen Ai',29:'Space',30:'Sports',31:'Super Power',32:'Vampire',33:'Yaoi',34:'Yuri',35:'Harem',36:'Slice of Life',37:'Supernatural',38:'Military', 39:'Police',40:'Psychological',41:'Thriller',42:'Seinen',43:'Josei'}

penalize_words_list = ['game', 'gameplay']

#Hyperparameters
min_dfVal, max_dfVal, kVal = 5, .85, 200
default_weight = 1
penalize_weight = 1

debug = False

def createModel(file):
    with open(file) as f:
            raw_docs = json.loads(f.readlines()[0])

    documents = []
    for anime in raw_docs["shows"]:
        reviews = ""
        for review in anime['reviews']:
            reviews += review['content']
        documents.append( (anime['title'], anime['description'], reviews, anime['image_url'], anime['promo_url'], anime['mal_id'], anime['rating'], anime['number_eps'], [genre_dict[x] for x in anime["genres"]], [x if x != "" else "No Studio Found" for x in anime["studios"]])   )

    return documents

documents = createModel('.'+os.path.sep+'anime_data1.json')
print("JSON Loaded", len(documents))

word_to_index, words_compressed, docs_compressed = None, None, None

if debug:
    vectorizer = TfidfVectorizer(stop_words = 'english', max_df = max_dfVal, min_df = min_dfVal)
    my_matrix = vectorizer.fit_transform([x[2] for x in documents]).transpose()
    words_compressed, _, docs_compressed = svds(my_matrix, k=kVal) 
    docs_compressed = docs_compressed.transpose()
    word_to_index = vectorizer.vocabulary_

    with open('word_to_index.pkl', 'wb') as f:
        pickle.dump(word_to_index, f)
    np.save('docs_compressed', docs_compressed)  
    np.save('words_compressed', words_compressed)  
else:
    with open('word_to_index.pkl', 'rb') as f:
        word_to_index = pickle.load(f)
    words_compressed = np.load('words_compressed.npy')
    docs_compressed = np.load('docs_compressed.npy')

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

print("Model Trained")

def readGames(file):
    with open(file) as f:
        raw_docs = json.loads(f.readlines()[0])
    documents = []
    for game in raw_docs:
        documents.append(raw_docs[game]['name'])

    return documents


print("Games File Loaded")
#Get list of game names for autocomplete
autocompleteGamesList = readGames('.' + os.path.sep + 'gamesList.json');

def getGames():
    url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
    r = requests.get(url)
    data = r.json()
    appList = data["applist"]["apps"]

    tupleList = np.array([(0, "")] * len(appList), dtype='object')
    for i, app in enumerate(appList):
        tupleList[i] = tuple([int(app['appid']), str(app['name'])])

    return tupleList

gameList = None
while True:
    gameList = getGames() #Run this only once please
    if len(gameList) > 0:
        print("gameList Loaded", len(gameList))
        break
    else:
        print("Gamelist is NOT populated, trying again", len(gameList))

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

def getGamesDescription(id):
    url = "https://store.steampowered.com/api/appdetails?appids=" + str(id)
    r = requests.get(url)
    data = r.json()
    if (data[str(id)]['success']):
        if data[str(id)]['data']['type'] == 'game':
            return remove_tags(data[str(id)]['data']['detailed_description'])
        else:
            return "Not Valid"
    else:
        return "Not Valid"
        
def getGameTags(id):
    url = "https://steamspy.com/api.php?request=appdetails&appid=" + str(id)
    r = requests.get(url)
    print(r)
    data = r.json()
    return data["tags"].keys()[:3]

def getAnimeList(game, gameList, id=False):
    desc = ""
    gameName = ""
    tags = ""
    if id:
        desc = getGamesDescription(game)
        gameName = "You entered the ID so you know this"
    else:
        gameIDs = getSimilarNames(gameList, game)
        #print(gameIDs)
        if len(gameIDs) == 0:
            return "No Game Found", "No Game Name"
        else:
            for ID in gameIDs:
                output = getGamesDescription(ID[0])
                if output == "Not Valid":
                    continue
                else:
                    desc = output
                    gameName = ID[1]
                    #tags = getGameTags(ID[0])
                    break
    
    if desc == "":
        return "No Game Found", "No Game Name"
        
    animeList = []
    animeCount = []
    
    #Tokenize the Description
    desc = desc.lower().split()
    
    for word in desc:
        weight = default_weight #Set weight
        if word.lower() in penalize_words_list: #If word in penalize list
            weight = penalize_weight #penalize the weight
            
        word_list = closest_project_to_word(word.lower(), 5) #Get Anime
        if word_list[0][0] != "Not in vocab.": #word_list[0][0]
            for anime in word_list: #for each anime in list of anime
                found = False
                for i, animeClosest in zip(range(len(animeList)), animeList): 
                    if animeClosest[0] == anime[0]: #found anime
                        animeList[i][1] += weight * anime[1] #Weight * Similarity
                        animeCount[i][1] += 1 #Count of Anime
                        found = True
                if not found:
                    animeList.append([anime[0], anime[1]])
                    animeCount.append([anime[0], 1])
    '''
    weight = 2
    for tag in tags:
        tag_words = closest_words(tag, 5)
        if tag_words[0][0] != "Not in Vocab":
            for word in tag_words:
                word_list = closest_project_to_word(word.lower(), 5)
                if word_list[0][0] != "Not in vocab.":
                    for anime in word_list: #for each anime in list of anime
                        found = False
                        for i, animeClosest in zip(range(len(animeList)), animeList): 
                            if animeClosest[0] == anime[0]: #found anime
                                animeList[i][1] += weight*anime[1]
                                found = True
                        if not found:
                            animeList.append([anime[0], weight*anime[1]])
    '''  
    #print(animeList)
    #for i, anime in zip(range(len(animeList)), animeList): 
    #    anime[1] = anime[1] / animeCount[i][1] 
    #print(animeList)
    
    weighting = max([x[1] for x in animeCount])
    
    final_list = sorted(animeList, key = lambda x: float(x[1]), reverse = True)
    final_anime = [x[0] for x in final_list]
    final_scores = [x[1]/weighting for x in final_list]
            
    return final_anime[:5], gameName, final_scores[:5]

def getAnimeInfo(AnimeName, AnimeScore):
    record = []
    for anime in documents:
        if AnimeName == anime[0]:
            record = [anime[0], anime[1], anime[3].split('?')[0], anime[4].split('?')[0], anime[5] , anime[6], anime[7], anime[8], anime[9], str(round(AnimeScore*100,2)) + "%"]
            break
    return record
    
print("All methods and data has been loaded sucessfully:")
print("JSON Anime:", len(documents))
print("Steam Games:", len(gameList))

@irsystem.route('/', methods=['GET'])
def search():
    query = request.args.get('search')
    if not query:
        data = []
        output_message = ''
    else:
        try:
            closestAnime, gameName, animeSimScores = getAnimeList(query, gameList)
            output_message = gameName
            
            if closestAnime == "No Game Found":
                data = []
                output_message = "Could not find game on Steam"
            else:
                info_anime = []
                for anime, score in zip(closestAnime, animeSimScores):
                    info_anime.append(getAnimeInfo(anime, score))
                
                #Logs
                print("USER QUERY =", query)
                print("RETURNED:", [anime[0] for anime in info_anime])

                data = []
                for anime in info_anime:
                    data.append(dict(name=anime[0],description=anime[1],picture=anime[2],video=anime[3],website="https://myanimelist.net/anime/"+str(anime[4]),rating=anime[5],eps=anime[6],genre=anime[7],studio=anime[8],simscore=anime[9]))
        except:
            print("Unexpected error:", sys.exc_info())
            data = []
            output_message = "Something went wrong, try another query"

    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data, game_list=autocompleteGamesList)



