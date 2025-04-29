import requests

url = "http://localhost:8000/predict"

payload = {
    "longitude": -118.28,
    "latitude": 37.57,
    "housing_median_age": 20,
    "total_rooms": 1000,
    "total_bedrooms": 500.0,
    "population": 3000,
    "households": 1000,
    "median_income": 5.5
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
