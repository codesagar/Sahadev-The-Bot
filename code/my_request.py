# USAGE
# python simple_request.py

# import the necessary packages
import requests
import pandas as pd

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
DATA_PATH = "tiny-dev-test.json"

# load the input data and construct the payload for the request
# ticket = 'INSOFE is working on developing a video surveillance tool with enhanced smart capabilities. The tool identifies the violation and sends out instant automated response without requiring any manual interference. Since the current process involves manually going through the footage and checking for violations, it is not only a time-consuming process but also requires manual hours and effort. The tool makes the entire process automated with an Embedded Machine Learning chip Question-- What is the paragraph talking about'
ticket = 'INSOFE has awarded over Rs. 3.2 Crores in merit scholarships in the last 2 years alone. INSOFE recognizes top performers and rewards them for demonstrating outstanding achievement at every phase of the program based on their performance and eligibility criteria. At each phase of the program, top performers are awarded rankings based on which scholarship winners are announced. Top performers can potentially win scholarships ranging from Rs. 25,000 to entire program fee and this can be attained on the successful completion of the program. Question-- What is the criteria for scholarship?'
payload = {"ticket": ticket}

# submit the request
r = requests.post(KERAS_REST_API_URL, data=payload).json()
print(ticket)
print(r)
# ensure the request was sucessful
if r["success"]:
	print('Hello')
    # loop over the predictions and display them
    # pred_df = pd.DataFrame(r['predictions'][0])
    # pred_df.to_csv('new_data_predictions.csv', index=False)

# otherwise, the request failed
else:
    print("Request failed")