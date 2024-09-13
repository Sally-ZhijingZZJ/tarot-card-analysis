import joblib

svc_model = joblib.load('svc_pipeline_model.pkl')

text = "how to resolve a challenge I'm facing in a relationship?"
prediction = svc_model.predict([text])

print(prediction[0])
