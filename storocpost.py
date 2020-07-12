import requests
occupancy = 11

data = {'unique_id' : 'ChIJ82TJ8MaxPIgRGd8xSBhWo54',
        'current_occupancy' : occupancy}

r = requests.post(url = API_ENDPOINT, json=data)

print(r.text)
