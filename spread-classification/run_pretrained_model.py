import joblib

svc_model = joblib.load('svc_pipeline_model.pkl')

text = "what will my life behave next year?"
prediction = svc_model.predict([text])

print(prediction[0])
