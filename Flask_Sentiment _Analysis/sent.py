import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# ✅ Load the TF-IDF vectorizer
with open("tfidf1_tweets.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# ✅ Load the Naive Bayes classifier
with open("NaiveBayes_Tweets.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    tweet = ""
    if request.method == "POST":
        tweet = request.form["tweet"]
        if tweet.strip():
            X = vectorizer.transform([tweet])
            pred = model.predict(X)[0]
            prediction = f"Predicted Sentiment: {pred}"
        else:
            prediction = "⚠️ Please enter a tweet."
    return render_template("sentiment.html", prediction=prediction, tweet=tweet)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
