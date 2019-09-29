import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

def predict(team1, team2, venue, toss_winner, toss_decision):
	matches = pd.read_csv('matches_modified.csv')
	matches = matches[pd.notnull(matches['winner'])]
	matches['city'].fillna('Dubai',inplace=True)

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
	
	t1scda = y.loc[((y.team1 == dicVal[team1]) & (y.team2 == dicVal[team2]) & (y.toss_winner == dicVal[toss_winner]) & (y.venue == le.transform([venue])[0]) ), 't1cs']
	t2scda = y.loc[((y.team1 == dicVal[team1]) & (y.team2 == dicVal[team2]) & (y.toss_winner == dicVal[toss_winner]) & (y.venue == le.transform([venue])[0]) ), 't2cs'] 
	
	if list(t1scda):
		t1scda = list(t1scda)[0]
	else:
		t1scda = 0

	if list(t2scda):
		t2scda = list(t2scda)[0]
	else:
		t2scda = 0

	model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
									max_depth=None, max_features='sqrt', max_leaf_nodes=None,
									min_impurity_decrease=0.0, min_impurity_split=None,
									min_samples_leaf=1, min_samples_split=2,
									min_weight_fraction_leaf=0.0, n_estimators=200,
									n_jobs=None, oob_score=False, random_state=None,
									verbose=0, warm_start=False)

	classification_model(model, x, predictor_var, outcome_var)

	predictions = model.predict(x_test)
	print(metrics.accuracy_score(y_test, predictions)*100)

	input = [dicVal[team1], dicVal[team2], le.transform([venue])[0], dicVal[toss_winner], toss_decision, t1scda, t2scda]
	input = np.array(input).reshape((1, -1))
	output = model.predict(input)
		#print(team1)
	return list(dicVal.keys())[list(dicVal.values()).index(output)], metrics.accuracy_score(y_test, predictions)*100

def classification_model(model, data, predictors, outcome):
	model.fit(data[predictors], data[outcome].values.ravel())
	kf = KFold(10, shuffle=True)
	error = []
	for train, test in kf.split(data):
		train_predictors = (data[predictors].iloc[train,:])
		train_target = data[outcome].iloc[train]
		model.fit(train_predictors, train_target.values.ravel())
		error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test].values.ravel()))

	print('Cross-Validation Score : %s' % '{0:.3%}'.format(np.mean(error)))
	model.fit(data[predictors], data[outcome].values.ravel())

def convert_back_to_ground_no(venue):
	venue_no = 0
	if venue == 'ACA-VDCA Stadium':
		venue_no =0
	if venue == 'Barabati Stadium':
		venue_no =1
	if venue ==  'Brabourne Stadium':
		venue_no =2
	if venue ==  'Buffalo Park':
		venue_no =3
	if venue ==  'De Beers Diamond Oval':
		venue_no =4
	if venue ==  'Dr DY Patil Sports Academy':
		venue_no =5
	if venue ==  'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium':
		venue_no =6
	if venue ==  'Dubai International Cricket Stadium':
		venue_no =7
	if venue ==  'Eden Gardens':
		venue_no =8
	if venue ==  'Feroz Shah Kotla':
		venue_no =9
	if venue ==  'Feroz Shah Kotla Ground':
		venue_no =10
	if venue ==  'Green Park':
		venue_no =11
	if venue ==  'Himachal Pradesh Cricket Association Stadium':
		venue_no =12
	if venue ==  'Holkar Cricket Stadium':
		venue_no =13
	if venue ==  'IS Bindra Stadium':
		venue_no =14
	if venue ==  'JSCA International Stadium Complex':
		venue_no =15
	if venue ==  'Kingsmead':
		venue_no =16
	if venue ==  'M Chinnaswamy Stadium':
		venue_no =17
	if venue ==  'M. A. Chidambaram Stadium':
		venue_no =18
	if venue ==  'M. Chinnaswamy Stadium':
		venue_no =19
	if venue ==  'MA Chidambaram Stadium, Chepauk':
		venue_no =20
	if venue ==  'Maharashtra Cricket Association Stadium':
		venue_no =21
	if venue ==  'Nehru Stadium':
		venue_no =22
	if venue ==  'New Wanderers Stadium':
		venue_no =23
	if venue ==  'Newlands':
		venue_no =24
	if venue ==  'OUTsurance Oval':
		venue_no =25
	if venue ==  'Punjab Cricket Association IS Bindra Stadium, Mohali':
		venue_no =26
	if venue ==  'Punjab Cricket Association Stadium, Mohali':
		venue_no =27
	if venue ==  'Rajiv Gandhi International Stadium, Uppal':
		venue_no =28
	if venue ==  'Rajiv Gandhi Intl. Cricket Stadium':
		venue_no =29
	if venue ==  'Sardar Patel Stadium, Motera':
		venue_no =30
	if venue ==  'Saurashtra Cricket Association Stadium':
		venue_no =31
	if venue ==  'Sawai Mansingh Stadium':
		venue_no =32
	if venue ==  'Shaheed Veer Narayan Singh International Stadium':
		venue_no =33
	if venue ==  'Sharjah Cricket Stadium':
		venue_no =34
	if venue ==  'Sheikh Zayed Stadium':
		venue_no =35
	if venue ==  "St George's Park":
		venue_no =36
	if venue ==  'Subrata Roy Sahara Stadium':
		venue_no =37
	if venue ==  'SuperSport Park':
		venue_no =38
	if venue ==  'Vidarbha Cricket Association Stadium, Jamtha':
		venue_no =39
	if venue ==  'Wankhede Stadium':
		venue_no =40
	return venue_no 
