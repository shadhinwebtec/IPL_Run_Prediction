from flask import Flask,render_template,request
import pickle
import numpy as np 

#open save model 
filename='IPL_target_runPrediction_model.pkl'
regression=pickle.load(open(filename,'rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])

def predict():
    if request.method == 'POST':
        # Mapping the team
        team_mapping = {
            'Chennai Super Kings': [1, 0, 0, 0, 0, 0, 0, 0],
            'Delhi Daredevils': [0, 1, 0, 0, 0, 0, 0, 0],
            'Kings XI Punjab': [0, 0, 1, 0, 0, 0, 0, 0],
            'Kolkata Knight Riders': [0, 0, 0, 1, 0, 0, 0, 0],
            'Mumbai Indians': [0, 0, 0, 0, 1, 0, 0, 0],
            'Rajasthan Royals': [0, 0, 0, 0, 0, 1, 0, 0],
            'Royal Challengers Bangalore': [0, 0, 0, 0, 0, 0, 1, 0],
            'Sunrisers Hyderabad': [0, 0, 0, 0, 0, 0, 0, 1]
        }
        
        temp_array = []

        # Extract boatting and bowling data
        batting_team = request.form['batting-team']
        bowling_team = request.form['bowling-team']

        # update temp arry via team
        temp_array.extend(team_mapping.get(batting_team, [0, 0, 0, 0, 0, 0, 0, 0]))
        temp_array.extend(team_mapping.get(bowling_team, [0, 0, 0, 0, 0, 0, 0, 0]))

        # Extract the other feature
        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        runs_in_prev_5_over = int(request.form['runs_in_prev_5_over'])
        wickets_in_prev_5_over = int(request.form['wickets_in_prev_5_over'])

        # Update temp_array with other features
        temp_array=temp_array+[overs, runs, wickets, runs_in_prev_5_over, wickets_in_prev_5_over]
        
        #data stored in array
        data = np.array([temp_array]) 
        my_prediction = int(regression.predict(data)[0])

        return render_template('result.html', lower_limit=my_prediction - 10, upper_limit=my_prediction + 5)


 
if __name__=='__main__':
    app.run(debug=True)    
            
            
        