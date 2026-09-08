from flask import Flask, render_template, url_for, request
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Stopwords & Stemmer initialization
cachedStopWords = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])
ps = PorterStemmer()

def preprocess_string(string):
    if not string or not isinstance(string, str):
        return ""
    cleaned_str = re.sub(r'[^a-z\s]+', ' ', string, flags=re.IGNORECASE).lower()
    words = [ps.stem(w) for w in cleaned_str.split() if w not in cachedStopWords]
    return ' '.join(words)

class MultinomialNaiveBayes:
    def __init__(self, unique_classes):
        self.classes = unique_classes

    def bagOfWords(self, headline, dict_index):
        if isinstance(headline, np.ndarray):
            headline = headline[0]
        for token_word in str(headline).split():
            self.bow_dicts[dict_index][token_word] += 1

    def train(self, dataset, labels):
        self.headline = dataset
        self.category = labels
        self.bow_dicts = np.array([defaultdict(lambda: 0) for _ in range(self.classes.shape[0])])

        if not isinstance(self.headline, np.ndarray):
            self.headline = np.array(self.headline)
        if not isinstance(self.category, np.ndarray):
            self.category = np.array(self.category)

        for cat_index, cat in enumerate(self.classes):
            all_cat_headline = self.headline[self.category == cat]
            cleaned_headline = [preprocess_string(cat_headline) for cat_headline in all_cat_headline]
            cleaned_headline = pd.DataFrame(data=cleaned_headline)
            np.apply_along_axis(self.bagOfWords, 1, cleaned_headline, cat_index)

        prob_classes = np.empty(self.classes.shape[0])
        all_words = []
        cat_word_counts = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            prob_classes[cat_index] = np.sum(self.category == cat) / float(self.category.shape[0])
            cat_word_counts[cat_index] = np.sum(np.array(list(self.bow_dicts[cat_index].values()))) + 1
            all_words += list(self.bow_dicts[cat_index].keys())

        self.vocab = np.unique(np.array(all_words))
        self.vocab_length = self.vocab.shape[0]

        denoms = np.array([cat_word_counts[cat_index] + self.vocab_length + 1 for cat_index, cat in enumerate(self.classes)])

        self.cats_info = [(self.bow_dicts[cat_index], prob_classes[cat_index], denoms[cat_index]) for cat_index, cat in enumerate(self.classes)]
        self.cats_info = np.array(self.cats_info, dtype=object)

    def getHeadlineProb(self, test_headline):
        likelihood_prob = np.zeros(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            for test_token in test_headline.split():
                test_token_counts = self.cats_info[cat_index][0].get(test_token, 0) + 1
                test_token_prob = test_token_counts / float(self.cats_info[cat_index][2])
                likelihood_prob[cat_index] += np.log(test_token_prob)

        post_prob = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            post_prob[cat_index] = likelihood_prob[cat_index] + np.log(self.cats_info[cat_index][1])

        return post_prob

    def test(self, test_set):
        predictions = []
        for headline in test_set:
            cleaned_headline = preprocess_string(headline)
            post_prob = self.getHeadlineProb(cleaned_headline)
            predictions.append(self.classes[np.argmax(post_prob)])
        return np.array(predictions)

# 4 Primary Category Metadata
CATEGORY_META = {
    1: {
        "id": 1,
        "name": "Sports",
        "icon": "⚽",
        "badge_class": "badge-sports",
        "template": "sports.html",
        "color": "#059669", # Emerald
        "description": "Athletic competitions, matches, leagues, scores, and tournament updates."
    },
    2: {
        "id": 2,
        "name": "Politics",
        "icon": "🏛️",
        "badge_class": "badge-politics",
        "template": "politics.html",
        "color": "#4F46E5", # Indigo
        "description": "Government affairs, elections, policy updates, legislation, and diplomacy."
    },
    3: {
        "id": 3,
        "name": "Entertainment",
        "icon": "🎬",
        "badge_class": "badge-entertainment",
        "template": "tv.html",
        "color": "#E11D48", # Rose
        "description": "Movies, television, celebrity culture, music release, awards, and arts."
    },
    5: {
        "id": 5,
        "name": "Technology",
        "icon": "💻",
        "badge_class": "badge-tech",
        "template": "tech.html",
        "color": "#0284C7", # Sky Blue
        "description": "Artificial intelligence, gadget launches, software development, startups, and innovation."
    }
}

# Global Model & Benchmark Accuracy initialization
MODEL = None
BENCHMARK_ACC = 0.92

def init_model():
    global MODEL, BENCHMARK_ACC
    print("Loading dataset news.csv and filtering for Sports, Politics, Entertainment, Technology...")
    df = pd.read_csv('news.csv', sep=',')
    
    # Filter dataset for only the 4 requested categories: 1 (Sports), 2 (Politics), 3 (Entertainment), 5 (Tech)
    df = df[df['Category'].isin([1, 2, 3, 5])].copy()
    
    y_train = np.asarray(df['Category'])
    x_train = np.asarray(df['Title'])

    # Evaluate benchmark accuracy on train/test split
    x_tr, x_te, y_tr, y_te = train_test_split(x_train, y_train, shuffle=True, test_size=0.25, random_state=42, stratify=y_train)
    eval_nb = MultinomialNaiveBayes(np.unique(y_tr))
    eval_nb.train(x_tr, y_tr)
    eval_preds = eval_nb.test(x_te)
    BENCHMARK_ACC = round(float(np.sum(eval_preds == y_te) / float(y_te.shape[0])), 4)

    # Train production model on full filtered dataset
    classes = np.unique(y_train)
    MODEL = MultinomialNaiveBayes(classes)
    MODEL.train(x_train, y_train)
    print(f"Model trained! 4-Category Benchmark Test Accuracy: {BENCHMARK_ACC * 100:.2f}%")

# Train model upon startup
init_model()


@app.route('/')
def home():
    return render_template('home.html', categories=CATEGORY_META)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('home.html', categories=CATEGORY_META)

    headline = request.form.get('message', '').strip()
    if not headline:
        return render_template('home.html', categories=CATEGORY_META, error="Please enter a news headline to classify.")

    cleaned_headline = preprocess_string(headline)
    log_probs = MODEL.getHeadlineProb(cleaned_headline)

    # Numerically stable Softmax calculation for confidence scores
    exp_p = np.exp(log_probs - np.max(log_probs))
    probs = exp_p / np.sum(exp_p)

    pred_idx = int(np.argmax(probs))
    pred_category_id = int(MODEL.classes[pred_idx])

    # Dynamic headline confidence score (%)
    headline_confidence = round(float(probs[pred_idx]) * 100, 1)

    meta = CATEGORY_META.get(pred_category_id, CATEGORY_META[1])

    payload = {
        "data": headline,
        "prediction_id": pred_category_id,
        "prediction": meta["name"],
        "meta": meta,
        "confidence": headline_confidence,
        "model_acc": round(BENCHMARK_ACC * 100, 1)
    }

    template_name = meta.get("template", "result.html")
    try:
        return render_template(template_name, **payload)
    except Exception:
        return render_template("result.html", **payload)


if __name__ == '__main__':
    app.run(debug=True)
