import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import json
import requests
import os

teamDictionary = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Leeds United FC": "Leeds",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Norwich City FC": "Norwich",
    "Southampton FC": "Southampton",
    "Tottenham Hotspur FC": "Tottenham",
    "Watford FC": "Watford",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    # Bundesliga
    "FC Augsburg": "Augsburg",
    "FC Bayern München": "Bayern Munich",
    "Arminia Bielefeld": "Bielefeld",
    "VfL Bochum 1848": "Bochum",
    "Borussia Dortmund": "Dortmund",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "1. FC Köln": "FC Koln",
    "SC Freiburg": "Freiburg",
    "SpVgg Greuther Fürth 1903": "Greuther Furth",
    "Hertha BSC": "Hertha",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Borussia Mönchengladbach": "M'gladbach",
    "1. FSV Mainz 05": "Mainz",
    "RB Leipzig": "RB Leipzig",
    "VfB Stuttgart": "Stuttgart",
    "1. FC Union Berlin": "Union Berlin",
    "VfL Wolfsburg": "Wolfsburg",
    # Ligue Une
    "Angers SCO": "Angers",
    "FC Girondins de Bordeaux": "Bordeaux",
    "Stade Brestois 29": "Brest",
    "Clermont Foot 63": "Clermont",
    "Racing Club de Lens": "Lens",
    "Lille OSC": "Lille",
    "FC Lorient": "Lorient",
    "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille",
    "FC Metz": "Metz",
    "AS Monaco FC": "Monaco",
    "Montpellier HSC": "Montpellier",
    "FC Nantes": "Nantes",
    "OGC Nice": "Nice",
    "Paris Saint-Germain FC": "Paris SG",
    "Stade de Reims": "Reims",
    "Stade Rennais FC 1901": "Rennes",
    "AS Saint-Étienne": "St Etienne",
    "RC Strasbourg Alsace": "Strasbourg",
    "ES Troyes AC": "Troyes",
    # La Liga
    "Deportivo Alavés": "Alaves",
    "Athletic Club": "Ath Bilbao",
    "Club Atlético de Madrid": "Ath Madrid",
    "FC Barcelona" : "Barcelona",
    "Real Betis Balompié" : "Betis",
    "Cádiz CF" : "Cadiz",
    "RC Celta de Vigo" : "Celta",
    "Elche CF" : "Elche",
    "RCD Espanyol de Barcelona": "Espanol",
    "Getafe CF" : "Getafe",
    "Granada CF" : "Granada",
    "Levante UD" : "Levante",
    "RCD Mallorca": "Mallorca",
    "CA Osasuna" : "Osasuna",
    "Real Madrid CF" : "Real Madrid",
    "Sevilla FC" : "Sevilla",
    "Real Sociedad de Fútbol" : "Sociedad",
    "Valencia CF" : "Valencia",
    "Rayo Vallecano de Madrid": "Vallecano",
    "Villarreal CF" : "Villarreal",
    # Serie A
    "Atalanta BC" : "Atalanta",
    "Bologna FC 1909" : "Bologna",
    "Cagliari Calcio" : "Cagliari",
    "Empoli FC" : "Empoli",
    "ACF Fiorentina" : "Fiorentina",
    "Genoa CFC" : "Genoa",
    "FC Internazionale Milano" : "Inter",
    "Juventus FC" : "Juventus",
    "SS Lazio" : "Lazio",
    "AC Milan" : "Milan",
    "SSC Napoli": "Napoli",
    "AS Roma" : "Roma",
    "US Salernitana 1919" : "Salernitana",
    "UC Sampdoria" : "Sampdoria",
    "US Sassuolo Calcio" : "Sassuolo",
    "Spezia Calcio" : "Spezia",
    "Torino FC" : "Torino",
    "Udinese Calcio" : "Udinese",
    "Venezia FC" : "Venezia",
    "Hellas Verona FC" : "Verona"
}

def downloadstats(url):
    return pd.read_csv(url, usecols=['Date', 'Time', 'HomeTeam', 'AwayTeam','FTHG','FTAG'])


def downloadfixures(league, dateFrom, dateTo):
    xAuthToken = os.environ.get('FootballPredictor')
    headers_dict = {"X-Auth-Token": xAuthToken}
    apiCall = "https://api.football-data.org/v2/competitions/" + league + "/matches?dateFrom=" + dateFrom + "&dateTo=" + dateTo
    response = json.loads(
                json.dumps(requests.get(
        apiCall,
        headers=headers_dict
    ).json()))

    matches = response["matches"]
    formattedMatches = []
    for match in matches:
        formattedMatches.append([teamDictionary[match["homeTeam"]["name"]], teamDictionary[match["awayTeam"]["name"]]])

    return formattedMatches


def getCoefficient(url):
    df = downloadstats(url)
    df['goal_difference'] = df['FTHG'] - df['FTAG']

    # create new variables to show home team win, draw or loss result
    df['home_win'] = np.where(df['goal_difference'] > 0, 1, 0)
    df['draw'] = np.where(df['goal_difference'] == 0, 1, 0)
    df['home_loss'] = np.where(df['goal_difference'] < 0, 1, 0)

    df_away = pd.get_dummies(df['AwayTeam'], dtype=np.int64)
    df_home = pd.get_dummies(df['HomeTeam'], dtype=np.int64)

    # subtract home from away
    df_model = df_home.sub(df_away)
    df_model.fillna(0, inplace=True)
    df_model['goal_difference'] = df['goal_difference']

    df_train = df_model  # setting up a training object equal to the df_model, not necessary

    lr = Ridge(alpha=0.001)
    X = df_train.drop(['goal_difference'], axis=1)
    y = df_train['goal_difference']

    lr.fit(X, y)

    df_ratings = pd.DataFrame(data={'team': X.columns, 'rating': lr.coef_})
    return df_ratings


def getCoefByTeam(team, ratings):
    for index, row in ratings.iterrows():
        if row["team"] == team:
            return row["rating"]


def predict(home, away, ratings):
    home_coef = getCoefByTeam(home, ratings)
    away_coef = getCoefByTeam(away, ratings)

    if home_coef >= away_coef:
        return ["home win", home, home_coef - away_coef]
    else:
        return ["away win", away, away_coef - home_coef]


def predictWeekendFixtures(games, ratings):
    results = []
    for game in games:
        results.append(predict(game[0], game[1], ratings))

    return results


def printResults(results):
    print()
    print("Results")
    print("========")
    for result in results:
        print(result[0] + " " + result[1] + " by: " + str(round(result[2],2)))


if __name__ == '__main__':
    PREMIERLEAGUE_STATS = "https://www.football-data.co.uk/mmz4281/2122/E0.csv"
    LALIGA_STATS = "https://www.football-data.co.uk/mmz4281/2122/SP1.csv"
    SA_STATS = "https://www.football-data.co.uk/mmz4281/2122/I1.csv"
    BL_STATS = "https://www.football-data.co.uk/mmz4281/2122/D1.csv"
    FL_STATS = "https://www.football-data.co.uk/mmz4281/2122/F1.csv"

    pl_ratings = getCoefficient(PREMIERLEAGUE_STATS)
    ll_ratings = getCoefficient(LALIGA_STATS)
    sa_ratings = getCoefficient(SA_STATS)
    bl_ratings = getCoefficient(BL_STATS)
    fl_ratings = getCoefficient(FL_STATS)

    dateFrom = "2021-09-10"
    dateTo = "2021-09-14"

    # print(sa_ratings)

    print(dateFrom + " to " + dateTo)
    pl_games = downloadfixures("PL", dateFrom, dateTo)
    pl_results = predictWeekendFixtures(pl_games, pl_ratings)
    printResults(pl_results)

    ll_games = downloadfixures("PD", dateFrom, dateTo)
    ll_results = predictWeekendFixtures(ll_games, ll_ratings)
    printResults(ll_results)

    sa_games = downloadfixures("SA", dateFrom, dateTo)
    sa_results = predictWeekendFixtures(sa_games, sa_ratings)
    printResults(sa_results)

    bl_games = downloadfixures("BL1", dateFrom, dateTo)
    bl_results = predictWeekendFixtures(bl_games, bl_ratings)
    printResults(bl_results)

    fl_games = downloadfixures("FL1", dateFrom, dateTo)
    fl_results = predictWeekendFixtures(fl_games, fl_ratings)
    printResults(fl_results)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
