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
from flask import send_from_directory

project_name = "Games2Anime: Anime Recommendations Based on Game Preferences"
net_id = "Amrit Amar (aa792),  Carina Cheng (chc94), Alan Pascual (ap835), Jeffrey Yao (jy398), Wenjia Zhang (wz276)"
TAG_RE = re.compile(r'<[^>]+>')
genre_dict = {1: 'Action', 2: 'Adventure', 3: 'Cars',4: 'Comedy',5: 'Dementia',6: 'Demons',7: 'Mystery',8: 'Drama', 9: 'Ecchi',10:'Fantasy',11:'Game',12:'Hentai',13:'Historical',14:'Horror',15:'Kids',16:'Magic',17:'Martial Arts',18:'Mecha',19:'Music',20:'Parody',21:'Samurai',22:'Romance',23:'School',24:'Sci-Fi',25:'Shoujo',26:'Shoujo Ai',27:'Shounen',28:'Shounen Ai',29:'Space',30:'Sports',31:'Super Power',32:'Vampire',33:'Yaoi',34:'Yuri',35:'Harem',36:'Slice of Life',37:'Supernatural',38:'Military', 39:'Police',40:'Psychological',41:'Thriller',42:'Seinen',43:'Josei'}

penalize_words_list = ['game', 'gameplay']

#Hyperparameters
min_dfVal, max_dfVal, kVal = 5, .85, 200
default_weight = 1
penalize_weight = .6

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

    return raw_docs

steamGamesList = readGames('.' + os.path.sep + 'gamesList.json');
print("GameList File Loaded")

#Get list of game names for autocomplete
autocompleteGamesList = [steamGamesList[x]['name'] for x in steamGamesList.keys()]
print("Autocomplete Populated")

#Get all games using Steam API
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
#Set to True to reactivate and remove the reassignment of gameList
while False:
    gameList = getGames() #Run this only once please
    if len(gameList) > 0:
        print("gameList Loaded", len(gameList))
        break
    else:
        print("Gamelist is NOT populated, trying again", len(gameList))

gameList = [(int(x), steamGamesList[x]['name']) for x in steamGamesList.keys()]

#Find similar names in the game list
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

def getRandomGames(file):
    with open(file) as f:
        raw_docs = json.load(f)
    return raw_docs['top_games']

randomGamesList = getRandomGames('.'+os.path.sep+'top_games_list.json')

def remove_tags(text):
    return TAG_RE.sub('', text)

#Use Steam API to get game description
def getGamesDescription(id):
    url = "https://store.steampowered.com/api/appdetails?appids=" + str(id)
    r = requests.get(url)
    data = r.json()
    if (data[str(id)]['success']):
        if data[str(id)]['data']['type'] == 'game':
            #return remove_tags(data[str(id)]['data']['detailed_description'])
            return data[str(id)]['data']['short_description']
        else:
            return "Not Valid"
    else:
        return "Not Valid"

#Use Steam Spy to get get game tags
def getGameTags(id):
    url = "https://steamspy.com/api.php?request=appdetails&appid=" + str(id)
    r = requests.get(url)
    #print(r)
    data = r.json()
    return data["tags"].keys()[:3]

#Get most recent Steam Games
def getRecentSteamGames(steamID):
    url = "http://api.steampowered.com/IPlayerService/GetRecentlyPlayedGames/v0001/?key=29312C7491002C407BB6EC6AB7634995&steamid=" + steamID + "&format=json"
    r = requests.get(url)
    data = r.json()
    l = []
    for game in data["response"]["games"]:
        l.append(game['name'])
    return l

def getSteamID(steamID):
    try:
        if int(steamID).bit_length() == 63:
            return steamID
    except:
        url = "https://steamid.xyz/" + steamID
        r = requests.get(url)
        htmltext = r.text
        a = re.search(r"profiles\/([\d]+)\"", htmltext)
        return (a.group()[:-1]).split('/')[1]


#Main Method using SteamID
def getAnimeListSteam(steamGames, gameList):
    desc = ""
    gameID = ""
    gameName = ""
    gameLink = ""
    tags = ""

    for game in steamGames:
        gameIDs = getSimilarNames(gameList, game)
        if len(gameIDs) == 0:
            return "No Game Found", "No Game Name"
        else:
            for ID in gameIDs:
                #output = getGamesDescription(ID[0])
                output = remove_tags(steamGamesList[str(ID[0])]['desc'])
                if output == "Not Valid":
                    continue
                else:
                    desc += output + " "
                    gameName += game + " "
                    gameID += str(ID[0]) + " "
                    gameLink = "https://store.steampowered.com/app/" + str(ID[0])
                    break

    if desc == "":
        return "No Game Found", "No Game Name"
    print(desc)
    print(gameName)

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

    return final_anime[:5], final_scores[:5], gameName, gameLink, gameID


#Main method
def getAnimeList(game, gameList):
    desc = ""
    gameID = 0
    gameName = ""
    gameLink = ""
    tags = ""

    gameIDs = getSimilarNames(gameList, game)
    if len(gameIDs) == 0:
        return "No Game Found", "No Game Name"
    else:
        for ID in gameIDs:
            #output = getGamesDescription(ID[0])
            output = remove_tags(steamGamesList[str(ID[0])]['desc'])
            if output == "Not Valid":
                continue
            else:
                desc = output
                gameName = ID[1]
                gameID = str(ID[0])
                gameLink = "https://store.steampowered.com/app/" + str(ID[0])
                break

    if desc == "":
        return "No Game Found", "No Game Name"

    animeList = []
    animeCount = []
    anime2word = dict()

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
                        if word in anime2word[anime[0]]:
                            anime2word[anime[0]][word] += 1
                        else:
                            anime2word[anime[0]][word] = 1
                        found = True
                if not found:
                    animeList.append([anime[0], anime[1]])
                    animeCount.append([anime[0], 1])
                    anime2word[anime[0]] = dict()
                    anime2word[anime[0]][word] = 1
                    
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
    
    topKeywords = set()
    anime2weight = dict()
    for k1,v_ in anime2word.items():
        anime2weight[k1] = dict()
        sortedV_ = {k: v for k, v in sorted(v_.items(), key=lambda item: item[1], reverse=True)}
        for k2, v in sortedV_.items():
            anime2weight[k1][k2] = v / (sum([v for k,v in sortedV_.items()]))
            topKeywords.add(k2)
        
    anime2keywordWeights = [(x, anime2weight[x]) for x in final_anime[:5]]
    
    animeKeywords = []
    #topKeywords = set()
    for anime, score in anime2keywordWeights:
        al = []
        ak = dict()
        for word, prob in score.items():
            #topKeywords.add(word)
            ak = dict(keyword=word,score=round(prob*100,2))
            al.append(ak)
        animeKeywords.append(al)
            
    topKeywords = list(topKeywords)

    return final_anime[:5], final_scores[:5], gameName, gameLink, gameID, topKeywords, animeKeywords

def getAnimeInfo(AnimeName, AnimeScore, AnimeKeywords):
    record = []
    for anime in documents:
        if AnimeName == anime[0]:
            record = [anime[0], anime[1], anime[3].split('?')[0], anime[4].split('?')[0], anime[5] , anime[6], anime[7], anime[8], anime[9], round(AnimeScore*100,2), AnimeKeywords]
            break
    return record

print("All methods and data has been loaded sucessfully:")
print("JSON Anime:", len(documents))
print("Steam Games:", len(gameList))

@irsystem.route('/', methods=['GET'])
def search():
    query = request.args.get('search')
    steamID = request.args.get('steam-input')
    isRandom = request.args.get('random-input')
    if not query and not steamID:
        data = []
        output_message = dict(message="")
    else:
        try:
            if steamID: #Steam ID takes precedence
                steamID = getSteamID(steamID.strip())
                steamUserGames = getRecentSteamGames(steamID.strip())
                closestAnime, animeSimScores, gameName, gameLink, gameID, topKeywords, animeKeywords = getAnimeListSteam(steamUserGames, gameList)
                output_message = dict(message="Steam Profile",link="https://steamcommunity.com/profiles/" + steamID,desc="Your most recent games were: " + ", ".join(steamUserGames),topkwords=", ".join(topKeywords))
                #print(steamUserGames)
            else:
                closestAnime, animeSimScores, gameName, gameLink, gameID, topKeywords, animeKeywords = getAnimeList(query, gameList)
                output_message = dict(message=gameName,link=gameLink,desc=getGamesDescription(int(gameID)),genres=", ".join(steamGamesList[gameID]['genre']), topkwords=", ".join(topKeywords))
                
            #print(topKeywords)
            #print(animeKeywords)
                
            if closestAnime == "No Game Found":
                data = []
                output_message = dict(message="Could not find the game on Steam. Try another search!")
            else:
                info_anime = []
                for anime, score, kw in zip(closestAnime, animeSimScores, animeKeywords):
                    info_anime.append(getAnimeInfo(anime, score, kw))

                #Logs
                print("USER QUERY =", query)
                print("RETURNED:", [anime[0] for anime in info_anime])

                data = []
                for anime in info_anime:
                    print(anime[10])
                    data.append(dict(name=anime[0],description=anime[1],picture=anime[2],video=anime[3],website="https://myanimelist.net/anime/"+str(anime[4]),rating=anime[5],eps=anime[6],genre=anime[7],studio=anime[8],simscore=anime[9],keywords=anime[10]))
        except:
            print("Unexpected error:", sys.exc_info())
            data = []
            output_message = dict(message="Something went wrong! Try another search!")

    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data, game_list=autocompleteGamesList, random_list=randomGamesList)

@irsystem.route('/about')
def about():
    return send_from_directory('templates', 'about.html')
