"""Fit transparent CPU baselines using training data only."""
from collections import Counter
from .core import LABELS, metrics

def evaluate_baselines(train, evaluation):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    truth = [LABELS[r["label"]] for r in evaluation]
    majority = Counter(r["label"] for r in train).most_common(1)[0][0]
    classifier = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000, sublinear_tf=True),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    classifier.fit([r["text"] for r in train], [r["label"] for r in train])
    prediction = [LABELS[int(x)] for x in classifier.predict([r["text"] for r in evaluation])]
    return {"majority": metrics(truth, [LABELS[majority]]*len(truth)),
            "tfidf_logistic": metrics(truth, prediction)}
