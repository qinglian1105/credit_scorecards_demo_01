import numpy as np
import pickle
import pandas as pd


# The following python script is mainly from an excellent project: 
# https://github.com/Rian021102/credit-scoring-analysis
def get_points_map_dict(scorecards):
    # Initialize the dictionary
    points_map_dict = {}
    points_map_dict['Missing'] = {}
    unique_char = set(scorecards['Characteristic'])
    for char in unique_char:
        # Get the Attribute & WOE info for each characteristics
        current_data = (scorecards[scorecards['Characteristic'] == char][['Attribute', 'Points']])          
        # Get the mapping
        points_map_dict[char] = {}
        for idx in current_data.index:
            attribute = current_data.loc[idx, 'Attribute']
            points = current_data.loc[idx, 'Points']
            if attribute == 'Missing':
                points_map_dict['Missing'][char] = points
            else:
                points_map_dict[char][attribute] = points
                points_map_dict['Missing'][char] = np.nan
    return points_map_dict


def transform_points(raw_data, points_map_dict, num_cols):
    points_data = raw_data.copy()
    # Map the data
    for col in points_data.columns:
        map_col = col  # No need to append '_bin' here
        points_data[col] = points_data[col].map(points_map_dict[map_col])
    # Map the data if there is a missing value or out of range value
    for col in points_data.columns:
        map_col = col  # No need to append '_bin' here
        points_data[col] = points_data[col].fillna(value=points_map_dict['Missing'][map_col])
    return points_data


# Predict the credit score
def predict_score(raw_data, points_map_dict, num_columns):
    # Transform raw input values into score points
    points = transform_points(raw_data = raw_data,
                              points_map_dict = points_map_dict,
                              num_cols = num_columns)    
    # print(points.T)
    # Caculate the score as the total points
    score = int(points.sum(axis=1))         
    reports = {'score': score}    
    if score < 250:
        reports['level'] = 'Very Poor'     
    elif score >= 250 and score < 300:
        reports['level'] = 'Poor'
    elif score >= 300 and score < 400:
        reports['level'] = 'Fair'
    elif score >= 400 and score < 500:
        reports['level'] = 'Good'
    elif score >= 500 and score < 600:
        reports['level'] = 'Very Good'
    elif score >= 600 and score < 700:
        reports['level'] = 'Exceptional'
    else:
        reports['level'] = 'Excellent'
    return reports


def inference(input):
    num_columns = ['annual_inc', 'loan_amnt', 'int_rate',]
    # load model
    scorecards = pd.read_pickle('models_prod/lendingclub.pkl')
    points_map_dict = get_points_map_dict(scorecards=scorecards)
    # print(points_map_dict)  
    input_table = pd.DataFrame(input, index=[0])    
    input_score = predict_score(raw_data = input_table, 
                                points_map_dict = points_map_dict, 
                                num_columns = num_columns)    
    return input_score


def main():
    # Your data
    inputs = {'annual_inc_bin': 160000, 
              'loan_amnt_bin': 10000, 
              'int_rate_bin': 6.58,        
              'purpose': 'car', 
              'grade': 'C', 
              'home_ownership': 'RENT', 
              'pub_rec_bankruptcies': 'N'}   
    # Inference and display results
    outputs = inference(inputs) 
    print("-"*28)
    print("Outputs: \n")
    print(outputs)
    print("-"*28)
    


if __name__ == "__main__":
    main()
