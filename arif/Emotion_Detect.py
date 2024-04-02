import joblib
import warnings
#can remove if u want its just to remove warning
warnings.filterwarnings("ignore", message="Trying to unpickle estimator .* from version .* when using version .*")

def predict_emotion(text):
    pipe_lr = joblib.load(open("./emotion_classifier_pipe_lr.pkl", "rb"))
    # Make prediction
    prediction = pipe_lr.predict([text])[0]
    return prediction
