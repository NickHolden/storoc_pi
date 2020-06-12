import requests
import json
response = requests.get("http://52.168.148.0/api")
decodedResponse = response.content
jsonResponse = json.loads(decodedResponse)

ID = jsonResponse["_id"]
sector = jsonResponse["sector"]
company = jsonResponse["company"]
uniqueID = jsonResponse["unique_id"]
address = jsonResponse["street_address"]

# get occupancy numbers from obj detection
data = {'unique_id': 1, 'current_occupancy': 27}                # requires unique_id and current_occupancy
print(decodedResponse)
post = requests.post("http://52.168.148.0/api", json=data)      # post to website
