'''
call this file after running app.py file to get prediction from server
'''
import requests

resp = requests.post('http://localhost:5000/predict', files={'file': open('pytorch_model_deployment/static/img/pop_ww.jpg', 'rb')})

print(resp.json())
