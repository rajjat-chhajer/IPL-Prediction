from prediction import predict , classification_model , convert_back_to_ground_no
from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'defaultkey'


@app.route('/', methods=['POST', 'GET'])
def get_data():
	return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def get():
	matches = pd.read_csv('matches_modified.csv')
	matches = matches[pd.notnull(matches['winner'])]
	matches['city'].fillna('Dubai',inplace=True)
	
	is_2019 = matches['season'] == 2019
	matches_2019 = matches[is_2019]
	match_result = []
	count = 0
	match_count = 0

	df2019 = pd.DataFrame(matches_2019)

	df2019.loc[df2019.team1 == "Delhi Capitals", 'team1' ] = "Delhi Daredevils"
	df2019.loc[df2019.team2 == "Delhi Capitals", 'team2' ] = "Delhi Daredevils"
	df2019.loc[df2019.toss_winner == "Delhi Capitals", 'toss_winner' ] = "Delhi Daredevils"
	df2019.loc[df2019.winner == "Delhi Capitals", 'winner' ] = "Delhi Daredevils"

	df2019.loc[df2019.team1 == "Rising Pune Supergiants", 'team1' ] = "Rising Pune Supergiant"
	df2019.loc[df2019.team2 == "Rising Pune Supergiants", 'team2' ] = "Rising Pune Supergiant"
	df2019.loc[df2019.toss_winner == "Rising Pune Supergiants", 'toss_winner' ] = "Rising Pune Supergiant"
	df2019.loc[df2019.winner == "Rising Pune Supergiants", 'winner' ] = "Rising Pune Supergiant"
	
	df2019.loc[df2019.venue == "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium", 'venue' ] = "ACA-VDCA Stadium"
	df2019.loc[df2019.venue == "Feroz Shah Kotla", 'venue' ] = "Feroz Shah Kotla Ground"
	df2019.loc[df2019.venue == "Punjab Cricket Association Stadium, Mohali", 'venue' ] = "IS Bindra Stadium"
	df2019.loc[df2019.venue == "Punjab Cricket Association IS Bindra Stadium, Mohali", 'venue' ] = "IS Bindra Stadium"
	df2019.loc[df2019.venue == "M Chinnaswamy Stadium", 'venue' ] = "M. Chinnaswamy Stadium"
	df2019.loc[df2019.venue == "M. A. Chidambaram Stadium", 'venue' ] = "MA Chidambaram Stadium, Chepauk"
	df2019.loc[df2019.venue == "Rajiv Gandhi International Stadium", 'venue' ] = "Rajiv Gandhi Intl. Cricket Stadium"
	df2019.apply(lambda x: sum(x.isnull()),axis=0)		

	matches2 = pd.DataFrame()
	matches = matches[['season','team1','team2','city','toss_decision','toss_winner','venue','t1cs','t2cs','winner']]

	matches2[['team1']] = matches[['team2']]
	matches2[['team2']] = matches[['team1']]
	matches2[['t1cs']] = matches[['t2cs']]
	matches2[['t2cs']] = matches[['t1cs']]
	matches2[['season','city','toss_decision','toss_winner','venue','winner']] = matches[['season','city','toss_decision','toss_winner','venue','winner']]
	matches = matches.append(matches2)

	matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
			 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
			 'Sunrisers Hyderabad','Rising Pune Supergiants','Rising Pune Supergiant','Kochi Tuskers Kerala','Pune Warriors','Delhi Capitals']
			,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','RPS','KTK','PW','DD'],inplace=True)

	encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
	  'team2': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
	  'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
	  'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13}}
	matches.replace(encode, inplace=True)

	matches.replace(['bat','field'], [1,2], inplace=True)

	dicVal = encode['winner']

	#removing duplicates
	df = pd.DataFrame(matches)
	
	df.loc[df.venue == "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium", 'venue' ] = "ACA-VDCA Stadium"
	df.loc[df.venue == "Feroz Shah Kotla", 'venue' ] = "Feroz Shah Kotla Ground"
	df.loc[df.venue == "Punjab Cricket Association Stadium, Mohali", 'venue' ] = "IS Bindra Stadium"
	df.loc[df.venue == "Punjab Cricket Association IS Bindra Stadium, Mohali", 'venue' ] = "IS Bindra Stadium"
	df.loc[df.venue == "M Chinnaswamy Stadium", 'venue' ] = "M. Chinnaswamy Stadium"
	df.loc[df.venue == "M. A. Chidambaram Stadium", 'venue' ] = "MA Chidambaram Stadium, Chepauk"
	df.loc[df.venue == "Rajiv Gandhi International Stadium", 'venue' ] = "Rajiv Gandhi Intl. Cricket Stadium"
	df.apply(lambda x: sum(x.isnull()),axis=0)		
	
	var_mod = ['city', 'venue']
	le = LabelEncoder()
	for i in var_mod:
		df[i] = le.fit_transform(df[i])
	
	#grid Search section
	x = df[df.season!=2019]
	y = df[df.season == 2019] 
	outcome_var = ['winner']
	predictor_var = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 't1cs', 't2cs']
	x_test = y[predictor_var]
	y_test = y[outcome_var]
	x_train = x[predictor_var]
	y_train = x[outcome_var]

	model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
									max_depth=None, max_features='sqrt', max_leaf_nodes=None,
									min_impurity_decrease=0.0, min_impurity_split=None,
									min_samples_leaf=1, min_samples_split=2,
									min_weight_fraction_leaf=0.0, n_estimators=200,
									n_jobs=None, oob_score=False, random_state=None,
									verbose=0, warm_start=False)

	classification_model(model, x, predictor_var, outcome_var)

	for index, row in df2019.iterrows():
		count += 1
		team1 = row['team1']
		team2 = row['team2']
		venue = row['venue']
		toss_winner = row['toss_winner']
		toss_decision = row['toss_decision']
		t1scda = row['t1cs']
		t2scda = row['t2cs']

		input = [dicVal[convert_back_to_team_names(team1)], dicVal[convert_back_to_team_names(team2)], le.transform([venue])[0], dicVal[convert_back_to_team_names(toss_winner)], tossDecision(toss_decision), t1scda, t2scda]
		input = np.array(input).reshape((1, -1))
		
		output = model.predict(input)
		winner = list(dicVal.keys())[list(dicVal.values()).index(output)]
		
		#winner = predict(convert_back_to_team_names(team1), convert_back_to_team_names(team2), venue, convert_back_to_team_names(toss_winner), tossDecision(toss_decision))

		winner_match = '<span class="danger">Not Matched</span>'
		if winner == convert_back_to_team_names(row['winner']):
			match_count += 1
			percentage = match_count/count
			winner_match = '<span class="success">Matched</span>'
		match_result.append([convert_back_to_team_names(team1), convert_back_to_team_names(team2), convert_back_to_team_names(row['winner']), winner, winner_match])
		
	return render_template('predict.html', result=match_result, count = count, match_count = match_count, percentage = percentage)

@app.route('/submit', methods=['POST', 'GET'])
def post():
	if request.method == 'POST':
		team1 = request.form['team1']
		team2 = request.form['team2']
		venue = request.form['venue']
		toss_winner = request.form['tossWinner']
		toss_decision = request.form['tossDecision']
		winner, accuracy = predict(convert_back_to_team_names(team1), convert_back_to_team_names(team2), venue, toss_winner, toss_decision)
		print("Winner -> " + winner)
		print("Accuracy -> " + str("{0:.2f}".format(accuracy)))
		#print("Winning Team is -> " + winner_team)
		#home_team_name = convert_back_to_team_names(team1).__str__()
		#away_team_name = convert_back_to_team_names(team2).__str__()
		#flash("Predicted Winner for this match is : " + winner)
		#return render_template('index.html')
		#return render_template('results.html')
		a = {'status':1, 'msg': "Predicted Winner: "+winner+'<br>' + "Model Accuracy: "+str("{0:.2f}".format(accuracy))+"%" }
		python2json = json.dumps(a)
		return python2json

def convert_back_to_team_names(team):
	team_name = ""

	if team == 'Kolkata' or team == 'Kolkata Knight Riders':
		team_name = "KKR"
	if team == "Bangalore" or team == 'Royal Challengers Bangalore':
		team_name = "RCB"
	if team == "Chennai" or team == 'Chennai Super Kings':
		team_name = "CSK"
	if team == "Jaipur" or team == 'Rajasthan Royals':
		team_name = "RR"
	if team == "Delhi" or team == 'Delhi Capitals' or team == 'Delhi Daredevils':
		team_name = "DD"
	if team == "Dharamshala" or team == 'Kings XI Punjab':
		team_name = "KXIP"
	if team == "Hyderabad" or team == 'Sunrisers Hyderabad':
		team_name = "SRH"
	if team == "Mumbai" or team == 'Mumbai Indians':
		team_name = "MI"
	if team == 'Deccan Chargers':
		team_name = "DC"
	if team == 'Gujarat Lions':
		team_name = "GL"
	if team == 'Rising Pune Supergiants' or team == 'Rising Pune Supergiant':
		team_name = "RPS"
	if team == 'Kochi Tuskers Kerala':
		team_name = "KT"
	if team == "Pune" or team == 'Pune Warriors':
		team_name = "PW"

	return team_name

def tossDecision(decision):
	dec = 0
	if decision == 'bat':
		dec = 1
	if decision == 'field':
		dec = 2

	return dec

if __name__ == '__main__':
	app.run()
