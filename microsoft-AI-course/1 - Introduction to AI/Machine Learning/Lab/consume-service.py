 
import urllib2
import json

data = {
        "Inputs": {
                "input1":
                [
                    {
                            'PatientID': "1",   
                            'Pregnancies': "1",   
                            'PlasmaGlucose': "1",   
                            'DiastolicBloodPressure': "1",   
                            'TricepsThickness': "1",   
                            'SerumInsulin': "1",   
                            'BMI': "1",   
                            'DiabetesPedigree': "1",   
                            'Age': "1",   
                    }
                ],
        },
    "GlobalParameters":  {
    }
}

body = str.encode(json.dumps(data))

url = 'https://europewest.services.azureml.net/subscriptions/2eaaf79985fe487bbf58b85fe1a8c92c/services/432851079c0947d689d04ba7ba01c70a/execute?api-version=2.0&format=swagger'
api_key = '4iQpDEIZSFXuRzcavc2IfAh9aGZ2F9WpWV9iuiSZGe0U2IHuov2dvqmlQBoxwlH+DJdTT3zhpKe+Jr+oqzSoxw==' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib2.Request(url, body, headers)

try:
    response = urllib2.urlopen(req)

    result = response.read()
    print(result)
except urllib2.HTTPError, error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read())) 
import urllib2
import json

data = {
        "Inputs": {
                "input1":
                [
                    {
                            'PatientID': "1",   
                            'Pregnancies': "1",   
                            'PlasmaGlucose': "1",   
                            'DiastolicBloodPressure': "1",   
                            'TricepsThickness': "1",   
                            'SerumInsulin': "1",   
                            'BMI': "1",   
                            'DiabetesPedigree': "1",   
                            'Age': "1",   
                    }
                ],
        },
    "GlobalParameters":  {
    }
}

body = str.encode(json.dumps(data))

url = 'https://europewest.services.azureml.net/subscriptions/2eaaf79985fe487bbf58b85fe1a8c92c/services/432851079c0947d689d04ba7ba01c70a/execute?api-version=2.0&format=swagger'
api_key = 'abc123' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib2.Request(url, body, headers)

try:
    response = urllib2.urlopen(req)

    result = response.read()
    print(result)
except urllib2.HTTPError, error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read())) 
