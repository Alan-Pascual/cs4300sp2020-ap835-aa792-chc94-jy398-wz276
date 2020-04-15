from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import numpy as np
import requests

project_name = "Culture for Quarantined Gamers: Anime Recommendations based on Game Preferences"
net_id = "Amrit Amar (aa792),  Carina Cheng (chc94), Alan Pascual (ap835), Jeffrey Yao (jy398), Wenjia Zhang (wz276)"

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ''
	else:
		output_message = "Your search: " + query + " (this will be replaced with anime)"
		data = range(5)
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)

######## Steam Functions ########
"""
	Calls Steam API to retrieve all public games' id and name.
		Returns: numpy array of tuples (appId : int, name : str)
"""
def getGames():
    url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
    r = requests.get(url)
    data = r.json()
    appList = data["applist"]["apps"]

    tupleList = np.array([(0, "")] * len(appList), dtype='object')
    for i, app in enumerate(appList):
        tupleList[i] = tuple([int(app['appid']), str(app['name'])])

	return tupleList

"""
	Compares query string with names of games in gamesList, and returns a list of
	similarly named game titles.
		TODO: Find a better similarity metric please
		Returns: numpy array of tuples (appId : int, name : str)
"""
def getSimilarNames(gamesList, query : str):
	similarNames = []
	for (appId, name) in gamesList:
		if query.lower() in name.lower():
			similarNames += [(appId, name)]
	return np.array(similarNames)

"""
	Retrieves reviews from given appId. Review is only collected if it has an 'up_vote'
	count above 3.
		Returns: numpy array of strings
"""
def getGameReviews(appId : int):
	url = "https://store.steampowered.com/appreviews/" + appId + "?json=1&num_per_page=100"
	r = requests.get(url)
    data = r.json()

	if data['success'] != 1:
		# Case: Game is unreleased, reviews are not open
		print('Please implement some warning for this')
	else:
		rawReviews = data['reviews']
		filteredReviews = np.array([], dtype = 'object')

		# TODO: Make condition to collect only good reviews
		for review in rawReviews:
	        if int(review['votes_up']) > 5:
				filteredReviews = np.append(filteredReviews, [review['review']])

		# TODO: Case when there are no/too little good reviews, should we just take the description?
		if filteredReviews.size <= 1:
			print('Btw, description will need a different API call')

		return filteredReviews

######## MAL Functions ########
