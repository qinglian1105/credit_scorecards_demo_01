# get_train_model.py
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os
import pandas as pd
import numpy as np
import psycopg2
import warnings

warnings.filterwarnings("ignore")


# Variables
ENV_FILE_PATH = ".env.development"
MODELS_DEV = "models_dev/lendingclub.pkl"
SQL_STR = """
    WITH tb01
    AS (
        SELECT CASE 
                WHEN loan_status LIKE ('%%Fully Paid%%')
                    THEN '0'
                WHEN loan_status LIKE ('%%Current%%')
                    THEN '0'
                WHEN loan_status LIKE ('%%Grace%%')
                    THEN '0'
                ELSE '1'
                END AS loan_status
            , annual_inc
            , loan_amnt
            , int_rate
            , purpose
            , grade
            , home_ownership
            , CASE 
                WHEN pub_rec_bankruptcies = 0
                    THEN 'N'
                ELSE 'Y'
                END AS pub_rec_bankruptcies
        FROM PUBLIC.lendingclub
        WHERE NOT (
                loan_status IS NULL
                OR annual_inc IS NULL
                OR loan_amnt IS NULL
                OR int_rate IS NULL
                OR purpose IS NULL
                OR grade IS NULL
                OR home_ownership IS NULL
                OR pub_rec_bankruptcies IS NULL
            )
            AND annual_inc >= 1000
            AND loan_amnt > 0
            AND home_ownership not in ('NONE','ANY')
    )
        , status_1
    AS (
        SELECT *
        FROM tb01
        WHERE tb01.loan_status IN ('1') LIMIT 200000
    )
        , status_0
    AS (
        SELECT *
        FROM tb01
        WHERE tb01.loan_status IN ('0') LIMIT 200000
    )
    SELECT *
    FROM status_1

    UNION ALL

    SELECT *
    FROM status_0
    ORDER BY int_rate ASC;
"""
num_clns = [
    "annual_inc",
    "loan_amnt",
    "int_rate",
]
cat_clns = ["purpose", "grade", "home_ownership", "pub_rec_bankruptcies"]
load_dotenv(ENV_FILE_PATH)
db_host = os.environ.get("PG_HOST")
db_name = os.environ.get("PG_DBNAME")
db_user = os.environ.get("PG_USER")
db_pwd = os.environ.get("PG_PASSWORD")
db_port = os.environ.get("PG_PORT")


# Functions
# Build DB Connection
def get_pg_conn(db_name, db_host, db_user, db_pwd, db_port):
    conn = psycopg2.connect(
        database=db_name, host=db_host, user=db_user, password=db_pwd, port=db_port
    )
    cur = conn.cursor()
    return conn, cur


# Create PG engine
def get_pg_engine(db_name, db_host, db_user, db_pwd, db_port):
    e = create_engine(
        f"postgresql://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}", echo=False
    )
    return e


# Get data from DB, make data cleaning, and display information
def get_pd_data(sql_str):
    conn = get_pg_engine(db_name, db_host, db_user, db_pwd, db_port)
    df = pd.read_sql(sql_str, conn, index_col=None)
    conn.dispose()
    df["loan_status"] = df["loan_status"].astype(int)
    df["annual_inc"] = df["annual_inc"].astype(int)
    df["loan_amnt"] = df["loan_amnt"].astype(int)
    df["int_rate"] = df["int_rate"].astype(float)
    df[cat_clns] = df[cat_clns].apply(lambda x: x.str.strip())
    df = df.dropna(axis=0)
    print("-" * 28)
    print("▲ Data size: \n\n", df.shape)
    print("-" * 28)
    print("▲ Data info: \n")
    print(df.info())
    print("-" * 28)
    print("▲ Data top 3: \n")
    print(df.head(3).T)
    return df


# The following python script is mainly from an excellent project:
# https://github.com/Rian021102/credit-scoring-analysis
# Create a function for binning the numerical predictor
def create_binning(data, predictor_label, num_of_bins):
    # Create a new column containing the binned predictor
    data[predictor_label + "_bin"] = pd.qcut(
        data[predictor_label].rank(method="first"), q=num_of_bins
    )
    return data


# Generate the WOE mapping dictionary
def get_woe_map_dict(WOE_table):
    # Initialize the dictionary
    WOE_map_dict = {}
    WOE_map_dict["Missing"] = {}
    unique_char = set(WOE_table["Characteristic"])
    for char in unique_char:
        # Get the Attribute & WOE info for each characteristics
        current_data = WOE_table[
            WOE_table["Characteristic"] == char
        ][  # Filter based on characteristic
            ["Attribute", "WOE"]
        ]  # Then select the attribute & WOE
        # Get the mapping
        WOE_map_dict[char] = {}
        for idx in current_data.index:
            attribute = current_data.loc[idx, "Attribute"]
            woe = current_data.loc[idx, "WOE"]
            if attribute == "Missing":
                WOE_map_dict["Missing"][char] = woe
            else:
                WOE_map_dict[char][attribute] = woe
                WOE_map_dict["Missing"][char] = np.nan
    # Validate data
    # print('Number of key : ', len(WOE_map_dict.keys()))
    # print(WOE_map_dict)
    return WOE_map_dict


# Replace the raw data in the train set with WOE values
def transform_woe(raw_data, WOE_dict, num_cols):
    woe_data = raw_data.copy()
    # Map the raw data
    for col in woe_data.columns:
        if col in num_cols:
            map_col = col + "_bin"
        else:
            map_col = col
        woe_data[col] = woe_data[col].map(WOE_dict[map_col])
    # Map the raw data if there is a missing value or out of range value
    for col in woe_data.columns:
        if col in num_cols:
            map_col = col + "_bin"
        else:
            map_col = col
        woe_data[col] = woe_data[col].fillna(value=WOE_dict["Missing"][map_col])
    return woe_data


def forward(X, y, predictors, scoring="roc_auc", cv=5):
    # Initialize list of results
    results = []
    # Define sample size and  number of all predictors
    n_samples, n_predictors = X.shape
    # Define list of all predictors
    col_list = np.arange(n_predictors)
    # Define remaining predictors for each k
    remaining_predictors = [p for p in col_list if p not in predictors]
    # Initialize list of predictors and its CV Score
    pred_list = []
    score_list = []
    # Cross validate each possible combination of remaining predictors
    for p in remaining_predictors:
        combi = predictors + [p]
        # Extract predictors combination
        X_ = X[:, combi]
        y_ = y
        # Define the estimator
        model = LogisticRegression(penalty=None, class_weight="balanced")
        # Cross validate the recall scores of the model
        cv_results = cross_validate(estimator=model, X=X_, y=y_, scoring=scoring, cv=cv)
        # Calculate the average CV/recall score
        score_ = np.mean(cv_results["test_score"])
        # Append predictors combination and its CV Score to the list
        pred_list.append(list(combi))
        score_list.append(score_)
    # Tabulate the results
    models = pd.DataFrame({"Predictors": pred_list, "Recall": score_list})
    # Choose the best model
    best_model = models.loc[models["Recall"].argmax()]
    return models, best_model


def create_factor_offset():
    # Define Factor and Offset
    factor = 80 / np.log(2)
    offset = 1000 - (factor * np.log(35))
    # print(f"Offset = {offset:.2f}")
    # print(f"Factor = {factor:.2f}")
    return factor, offset


def create_scorecards(
    factor,
    offset,
    forward_models,
    predictors,
    best_model,
    best_model_summary,
    WOE_table,
):
    num_columns = num_clns
    best_predictors = forward_models["Predictors"].loc[len(predictors)]
    # Define n = number of characteristics
    n = len(best_predictors)
    # Define b0
    b0 = best_model.intercept_[0]
    # print(f"n = {n}")
    # print(f"b0 = {b0:.4f}")
    # Adjust characteristic name in best_model_summary_table
    for col in best_model_summary["Characteristic"]:
        if col in num_columns:
            bin_col = col + "_bin"
        else:
            bin_col = col
        best_model_summary.replace(col, bin_col, inplace=True)
        # Merge tables to get beta_i for each characteristic
        scorecards = pd.merge(
            left=WOE_table, right=best_model_summary, how="left", on=["Characteristic"]
        )
    # print(scorecards.head())
    # Define beta and WOE
    beta = scorecards["Estimate"]
    WOE = scorecards["WOE"]
    # Calculate the score point for each attribute
    scorecards["Points"] = (offset / n) - factor * ((b0 / n) + (beta * WOE))
    scorecards["Points"] = scorecards["Points"].astype("int")
    # print(scorecards)
    return scorecards


def main():
    ds = get_pd_data(SQL_STR)
    data = ds.copy()
    response_variable = "loan_status"
    print("-" * 28)
    print(
        "Check the proportion of response variable: \n",
        data[response_variable].value_counts(normalize=True),
    )
    y = data[response_variable]
    X = data.drop(columns=[response_variable], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )
    data_train = pd.concat((X_train, y_train), axis=1)
    for column in num_clns:
        data_train_binned = create_binning(
            data=data_train, predictor_label=column, num_of_bins=4
        )
    # print(data_train_binned.T)
    # Define the initial empty list
    crosstab_num = []
    for column in num_clns:
        # Create a contingency table
        crosstab = pd.crosstab(
            data_train_binned[column + "_bin"],
            data_train_binned[response_variable],
            margins=True,
        )
        # Append to the list
        crosstab_num.append(crosstab)

    # Define the initial empty list
    crosstab_cat = []
    for column in cat_clns:
        # Create a contingency table
        crosstab = pd.crosstab(
            data_train_binned[column],
            data_train_binned[response_variable],
            margins=True,
        )
        # Append to the list
        crosstab_cat.append(crosstab)
    crosstab_list = crosstab_num + crosstab_cat
    # print("-"*28)
    # print(crosstab_list)

    # Define the initial list for WOE
    WOE_list = []
    # Define the initial list for IV
    IV_list = []
    # Create the initial table for IV
    IV_table = pd.DataFrame({"Characteristic": [], "Information Value": []})

    # Perform the algorithm for all crosstab
    for crosstab in crosstab_list:
        # Calculate % Good
        crosstab["p_good"] = crosstab[0] / crosstab[0]["All"]
        # Calculate % Bad
        crosstab["p_bad"] = crosstab[1] / crosstab[1]["All"]
        # Calculate the WOE
        crosstab["WOE"] = np.log(crosstab["p_good"] / crosstab["p_bad"])
        # Calculate the contribution value for IV
        crosstab["contribution"] = (crosstab["p_good"] - crosstab["p_bad"]) * crosstab[
            "WOE"
        ]
        # Calculate the IV
        IV = crosstab["contribution"][:-1].sum()
        add_IV = {"Characteristic": crosstab.index.name, "Information Value": IV}
        WOE_list.append(crosstab)
        IV_list.append(add_IV)
    # print("-"*58)
    # print(WOE_list)

    # Create initial table to summarize the WOE values
    WOE_table = pd.DataFrame({"Characteristic": [], "Attribute": [], "WOE": []})
    for i in range(len(crosstab_list)):
        # Define crosstab and reset index
        crosstab = crosstab_list[i].reset_index()
        # Save the characteristic name
        char_name = crosstab.columns[0]
        # Only use two columns (Attribute name and its WOE value)
        # Drop the last row (average/total WOE)
        crosstab = crosstab.iloc[:-1, [0, -2]]
        crosstab.columns = ["Attribute", "WOE"]
        # Add the characteristic name in a column
        crosstab["Characteristic"] = char_name
        WOE_table = pd.concat((WOE_table, crosstab), axis=0)
        # Reorder the column
        WOE_table.columns = ["Characteristic", "Attribute", "WOE"]
    # print("-"*58)
    # print(WOE_table)

    # Put all IV in the table
    IV_table = pd.DataFrame(IV_list)
    IV_table
    # print("-"*58)
    # print(IV_table)

    # Define the predictive power of each characteristic
    strength = []
    # Assign the rule of thumb regarding IV
    for iv in IV_table["Information Value"]:
        if iv < 0.02:
            strength.append("Unpredictive")
        elif iv >= 0.02 and iv < 0.1:
            strength.append("Weak")
        elif iv >= 0.1 and iv < 0.3:
            strength.append("Medium")
        elif iv >= 0.3 and iv < 0.5:
            strength.append("Strong")
        else:
            strength.append("Very strong")

    # Assign the strength to each characteristic
    IV_table = IV_table.assign(Strength=strength)

    # Sort the table by the IV values
    IV_table.sort_values(by="Information Value")
    # print("-"*58)
    # print(IV_table.sort_values(by='Information Value'))

    # Generate the WOE map dictionary
    WOE_map_dict = get_woe_map_dict(WOE_table=WOE_table)
    WOE_map_dict
    # print("-"*58)
    # print(WOE_map_dict)

    # Transform the X_train
    woe_train = transform_woe(
        raw_data=X_train, WOE_dict=WOE_map_dict, num_cols=num_clns
    )
    woe_train = woe_train.fillna(0)
    # print("-"*58)
    # print(woe_train)

    # Transform the X_test
    woe_test = transform_woe(raw_data=X_test, WOE_dict=WOE_map_dict, num_cols=num_clns)
    # print("-"*58)
    # print(woe_test)

    # Rename the raw X_train for the future
    raw_train = X_train
    # print(raw_train)
    # Define X_train
    X_train = woe_train.to_numpy()
    # print(X_train)
    # Check y_train
    y_train = y_train.to_numpy()
    # print(y_train)

    # Define predictor for the null model
    predictor = []
    # The predictor in the null model is zero values for all predictors
    X_null = np.zeros((X_train.shape[0], 1))
    # Define the estimator
    model = LogisticRegression(penalty=None, class_weight="balanced")
    # Cross validate
    cv_results = cross_validate(
        estimator=model, X=X_null, y=y_train, cv=10, scoring="recall"
    )

    # Calculate the average CV/recall score
    score_ = np.mean(cv_results["test_score"])
    # Create table for the best model of each k predictors
    # Append the results of null model
    forward_models = pd.DataFrame({"Predictors": [predictor], "Recall": [score_]})
    # print(forward_models)

    # Define list of predictors
    predictors = []
    n_predictors = X_train.shape[1]

    # Perform forward selection procedure for k=1,...,11 predictors
    for k in range(n_predictors):
        _, best_model = forward(
            X=X_train, y=y_train, predictors=predictors, scoring="recall", cv=10
        )

        # Tabulate the best model of each k predictors
        forward_models.loc[k + 1] = best_model
        predictors = best_model["Predictors"]
    # Display the results
    # print(forward_models)

    # Find the best Recall score
    best_idx = forward_models["Recall"].argmax()
    best_recall = forward_models["Recall"].loc[best_idx]
    best_predictors = forward_models["Predictors"].loc[best_idx]
    # Print the summary
    # print('Best index            :', best_idx)
    # print('Best Recall           :', best_recall)
    # print('Best predictors (idx) :', best_predictors)
    # print('Best predictors       :')
    # print(raw_train.columns[best_predictors].tolist())

    # Define X with best predictors
    X_train_best = X_train[:, best_predictors]

    # Fit best model
    best_model = LogisticRegression(penalty=None, class_weight="balanced")
    best_model.fit(X_train_best, y_train)
    best_model_intercept = pd.DataFrame(
        {"Estimate": best_model.intercept_}, index=["Intercept"]
    )
    # print(best_model_intercept)

    best_model_params = raw_train.columns[best_predictors].tolist()
    best_model_coefs = pd.DataFrame(
        {"Estimate": np.reshape(best_model.coef_, best_idx)}, index=best_model_params
    )
    best_model_summary = pd.concat((best_model_intercept, best_model_coefs), axis=0)
    # print("-"*28)
    # print(best_model_summary)
    # print("-"*28)

    # Predict class labels for sample in X_train.
    y_train_pred = best_model.predict(X_train_best)
    # print(y_train_pred)

    # Rename the raw X_test for the future
    # raw_test = X_test
    # Define X_test
    X_test = woe_test.to_numpy()
    # print(X_test)
    y_test = y_test.to_numpy()
    # print(y_test)

    # Define X_test with best predictors
    X_test_best = X_test[:, best_predictors]

    # Predict class labels for sample in X_test.
    y_test_pred = best_model.predict(X_test_best)
    # print(y_test_pred)

    # Calculate the recall score on the test set
    recall_test = recall_score(y_true=y_test, y_pred=y_test_pred)
    # print(recall_test)
    # Predict the probability estimates
    y_test_pred_proba = best_model.predict_proba(X_test_best)[:, [1]]
    # print(y_test_pred_proba)
    best_predictors = forward_models["Predictors"].loc[len(predictors)]
    # Define X with best predictors
    X_train_best = X_train[:, best_predictors]

    # Fit best model
    best_model = LogisticRegression(penalty=None, class_weight="balanced")
    best_model.fit(X_train_best, y_train)
    best_model_intercept = pd.DataFrame(
        {"Characteristic": "Intercept", "Estimate": best_model.intercept_}
    )
    # print(best_model_intercept)

    best_model_params = raw_train.columns[best_predictors].tolist()
    best_model_coefs = pd.DataFrame(
        {
            "Characteristic": best_model_params,
            "Estimate": np.reshape(best_model.coef_, len(best_predictors)),
        }
    )
    best_model_summary = pd.concat(
        (best_model_intercept, best_model_coefs), axis=0, ignore_index=True
    )
    # print(best_model_summary)

    factor, offset = create_factor_offset()
    scorecards = create_scorecards(
        factor,
        offset,
        forward_models,
        predictors,
        best_model,
        best_model_summary,
        WOE_table,
    )
    scorecards.to_pickle(MODELS_DEV)


if __name__ == "__main__":
    time_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    main()
    time_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("-" * 28)
    print("Successfully trained.")
    print(time_start)
    print(time_end)
    print("-" * 28)
