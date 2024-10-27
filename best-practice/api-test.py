import requests

insurance = {"smoker": "yes", "sex": "female", "children": 0, "bmi": 26.29, "age": 62}


url = 'http://localhost:9696/predict'
response = requests.post(url, json=insurance, timeout=10)
print(response.json())
