import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import json
import requests
import os

team_dictionary = {
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
    "FC Barcelona": "Barcelona",
    "Real Betis Balompié": "Betis",
    "Cádiz CF": "Cadiz",
    "RC Celta de Vigo": "Celta",
    "Elche CF": "Elche",
    "RCD Espanyol de Barcelona": "Espanol",
    "Getafe CF": "Getafe",
    "Granada CF": "Granada",
    "Levante UD": "Levante",
    "RCD Mallorca": "Mallorca",
    "CA Osasuna": "Osasuna",
    "Real Madrid CF": "Real Madrid",
    "Sevilla FC": "Sevilla",
    "Real Sociedad de Fútbol": "Sociedad",
    "Valencia CF": "Valencia",
    "Rayo Vallecano de Madrid": "Vallecano",
    "Villarreal CF": "Villarreal",
    # Serie A
    "Atalanta BC": "Atalanta",
    "Bologna FC 1909": "Bologna",
    "Cagliari Calcio": "Cagliari",
    "Empoli FC": "Empoli",
    "ACF Fiorentina": "Fiorentina",
    "Genoa CFC": "Genoa",
    "FC Internazionale Milano": "Inter",
    "Juventus FC": "Juventus",
    "SS Lazio": "Lazio",
    "AC Milan": "Milan",
    "SSC Napoli": "Napoli",
    "AS Roma": "Roma",
    "US Salernitana 1919": "Salernitana",
    "UC Sampdoria": "Sampdoria",
    "US Sassuolo Calcio": "Sassuolo",
    "Spezia Calcio": "Spezia",
    "Torino FC": "Torino",
    "Udinese Calcio": "Udinese",
    "Venezia FC": "Venezia",
    "Hellas Verona FC": "Verona"
}


def downloadstats(url):
    return pd.read_csv(url, usecols=['Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])


def downloadfixures(league, date_from, date_to):
    x_auth_token = os.environ.get('FootballPredictor')
    headers_dict = {"X-Auth-Token": x_auth_token}
    api_call = "https://api.football-data.org/v2/competitions/" + league + "/matches?dateFrom=" + date_from + "&dateTo=" + date_to
    response = json.loads(
                json.dumps(
                    requests.get(
                        api_call,
                        headers=headers_dict
                    ).json()))

    matches = response["matches"]
    formatted_matches = []
    for match in matches:
        formatted_matches.append([team_dictionary[match["homeTeam"]["name"]], team_dictionary[match["awayTeam"]["name"]]])

    return formatted_matches


def get_coefficient(url):
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
    x = df_train.drop(['goal_difference'], axis=1)
    y = df_train['goal_difference']

    lr.fit(x, y)

    df_ratings = pd.DataFrame(data={'team': x.columns, 'rating': lr.coef_})
    return df_ratings


def get_coefficient_by_team(team, ratings):
    for index, row in ratings.iterrows():
        if row["team"] == team:
            return row["rating"]


def predict(home, away, ratings):
    home_coef = get_coefficient_by_team(home, ratings)
    away_coef = get_coefficient_by_team(away, ratings)

    if home_coef >= away_coef:
        return ["home win", home, home_coef - away_coef]
    else:
        return ["away win", away, away_coef - home_coef]


def predict_weekend_fixtures(games, ratings):
    results = []
    for game in games:
        results.append(predict(game[0], game[1], ratings))

    return results


def print_results(results):
    print()
    print("Results")
    print("========")
    for result in results:
        print(result[0] + " " + result[1] + " by: " + str(round(result[2], 2)))


if __name__ == '__main__':
    PREMIERLEAGUE_STATS = "https://www.football-data.co.uk/mmz4281/2122/E0.csv"
    LALIGA_STATS = "https://www.football-data.co.uk/mmz4281/2122/SP1.csv"
    SA_STATS = "https://www.football-data.co.uk/mmz4281/2122/I1.csv"
    BL_STATS = "https://www.football-data.co.uk/mmz4281/2122/D1.csv"
    FL_STATS = "https://www.football-data.co.uk/mmz4281/2122/F1.csv"

    pl_ratings = get_coefficient(PREMIERLEAGUE_STATS)
    ll_ratings = get_coefficient(LALIGA_STATS)
    sa_ratings = get_coefficient(SA_STATS)
    bl_ratings = get_coefficient(BL_STATS)
    fl_ratings = get_coefficient(FL_STATS)

    dateFrom = "2021-09-17"
    dateTo = "2021-09-21"

    # print(sa_ratings)

    print(dateFrom + " to " + dateTo)
    pl_games = downloadfixures("PL", dateFrom, dateTo)
    pl_results = predict_weekend_fixtures(pl_games, pl_ratings)
    print_results(pl_results)

    ll_games = downloadfixures("PD", dateFrom, dateTo)
    ll_results = predict_weekend_fixtures(ll_games, ll_ratings)
    print_results(ll_results)

    sa_games = downloadfixures("SA", dateFrom, dateTo)
    sa_results = predict_weekend_fixtures(sa_games, sa_ratings)
    print_results(sa_results)

    bl_games = downloadfixures("BL1", dateFrom, dateTo)
    bl_results = predict_weekend_fixtures(bl_games, bl_ratings)
    print_results(bl_results)

    fl_games = downloadfixures("FL1", dateFrom, dateTo)
    fl_results = predict_weekend_fixtures(fl_games, fl_ratings)
    print_results(fl_results)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
