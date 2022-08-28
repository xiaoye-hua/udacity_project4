import json

import requests
import logging


logging.basicConfig(
    level=logging.INFO
)

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"
port = 9000



#Call each API endpoint and store the responses
# response1=requests.get('http://127.0.0.1:4000/prediction').content
response1 = requests.post(f'http://127.0.0.1:{port}/prediction?path=ingesteddata/finaldata.csv').content.decode('ascii') #put an API call here
logging.info(f"response1: {response1}")
response2 = requests.get(f'http://127.0.0.1:{port}/scoring').content.decode('ascii')#put an API call here
logging.info(f"response2: {response2}")

response3 = requests.post(
    f'http://127.0.0.1:{port}/summarystats?path=ingesteddata/finaldata.csv').content.decode('ascii')  # put an API call here
logging.info(f"response3: {response3}")
# response4 = #put an API call here

response4 = requests.post(
    f'http://127.0.0.1:{port}/diagnostics?path=ingesteddata/finaldata.csv').content.decode('ascii')  # put an API call here
logging.info(f"response4: {response4}")

# #combine all API responses
responses = {
    'res1': response1,
    'res2': response2,
    'res3': response3,
    'res4': response4
}

# print(responses)f

#write the responses to your workspace

with open('apireturns.txt', 'w') as f:
    json.dump(responses, f)



