"""
pipeline_model.py
-----------------
Custom transformers for the StyleSense recommendation pipeline.
NLP backend: spaCy en_core_web_sm (pretrained statistical model).

Components used from en_core_web_sm:
  tok2vec        — contextual token embeddings
  tagger         — statistical POS tagger     (token.pos_, token.tag_)
  parser         — dependency parser          (token.dep_, token.head, token.children)
  lemmatizer     — rule-assisted lemmatizer   (token.lemma_)
  ner            — statistical NER            (doc.ents)
  attribute_ruler— attribute mapping

Install the model:
    pip install en_core_web_sm-3.8.0-py3-none-any.whl

Run the dashboard:
    streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_FEATURES   = ['Age', 'Positive Feedback Count', 'Clothing ID']
CATEGORICAL_FEATURES = ['Division Name', 'Department Name', 'Class Name']

# ── Fashion-domain vocabulary for the EntityRuler (sits before statistical NER)
_CLOTHING_ITEMS = {
    'dress','blouse','jeans','skirt','jacket','top','shirt','pants','sweater',
    'cardigan','coat','blazer','shorts','leggings','romper','jumpsuit','vest',
    'tunic','hoodie','pullover','bra','tank','tee','dresses','blouses','tops',
    'shirts','jackets','coats','blazers','sweaters','cardigans','skirts',
}
_FIT_TERMS = {
    'small','large','tight','loose','fitted','baggy','oversized','snug',
    'petite','narrow','wide','short','long',
}
_MATERIAL_TERMS = {
    'cotton','linen','silk','polyester','wool','cashmere','spandex','lace',
    'knit','jersey','chiffon','satin','rayon','viscose','nylon','denim',
    'velvet','suede','leather','tweed',
}


def load_nlp():
    """
    Load en_core_web_sm and prepend a fashion-domain EntityRuler.

    The EntityRuler fires BEFORE the statistical NER so that clothing
    terminology (CLOTHING, FIT, MATERIAL, SENTIMENT_*) is captured even when
    the general-purpose NER would not recognise them.  Statistical NER still
    runs afterwards to catch any remaining entities (GPE, ORG, etc.).
    """
    nlp = spacy.load('en_core_web_sm')

    ruler = nlp.add_pipe('entity_ruler', before='ner')
    patterns = []

    for token in _CLOTHING_ITEMS:
        patterns.append({'label': 'CLOTHING', 'pattern': token})
    for token in _FIT_TERMS:
        patterns.append({'label': 'FIT',      'pattern': token})
    for token in _MATERIAL_TERMS:
        patterns.append({'label': 'MATERIAL', 'pattern': token})

    for phrase in ['runs small','runs large','true to size','fits true',
                   'too small','too large','size up','size down',
                   'runs true','fits perfectly','fits well']:
        patterns.append({'label': 'FIT', 'pattern': phrase})

    for w in ['love','perfect','beautiful','amazing','gorgeous','flattering',
              'comfortable','recommend','adorable','lovely','excellent','fantastic']:
        patterns.append({'label': 'SENTIMENT_POS', 'pattern': w})
    for w in ['disappointed','terrible','awful','cheap','ugly','uncomfortable',
              'waste','returning','returned','disappointing']:
        patterns.append({'label': 'SENTIMENT_NEG', 'pattern': w})

    ruler.add_patterns(patterns)
    return nlp


class TextCombiner(BaseEstimator, TransformerMixin):
    """Concatenates Title + Review Text into one string per row."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X['Title'].fillna('').astype(str) + ' ' +
                X['Review Text'].fillna('').astype(str)).str.strip()


class FullFeatureUnion(BaseEstimator, TransformerMixin):
    """
    Single-pass feature transformer using en_core_web_sm.

    Runs nlp.pipe() ONCE per fit/transform call and simultaneously extracts:

    1. Structured features
       - Numerical  : StandardScaler  (Age, Positive Feedback Count, Clothing ID)
       - Categorical : OneHotEncoder  (Division, Department, Class Name)

    2. TF-IDF text features
       Preprocessing via en_core_web_sm:
         • token.lemma_  — lemmatization (dresses→dress, recommended→recommend)
         • token.pos_    — keep only NOUN/VERB/ADJ/ADV (content-word filter)
         • token.is_stop — remove stop words
         • token.is_alpha — remove punctuation and numbers
       Then: TfidfVectorizer with unigrams + bigrams, sublinear TF scaling.

    3. spaCy NLP features  (20 per review)
       Derived from en_core_web_sm pipeline components:

       Tokenizer:
         [0]  char_count        total characters
         [1]  word_count        alpha token count
         [2]  sent_count        sentence count
         [3]  excl_count        exclamation marks
         [4]  quest_count       question marks
         [5]  avg_word_length   mean alpha token length

       Statistical POS tagger  (token.pos_):
         [6]  adj_count         adjective count  (POS=ADJ)
         [7]  adv_count         adverb count     (POS=ADV)
         [8]  verb_count        verb count       (POS=VERB)
         [9]  noun_count        noun count       (POS=NOUN)
         [10] adj_ratio         adj_count / word_count
         [11] adv_ratio         adv_count / word_count

       Dependency parser  (token.dep_, token.children):
         [12] negation_count    number of 'neg' dependency arcs
         [13] neg_adj_count     adjectives with a 'neg' child
                                ("not flattering" — parser links 'not'→'flattering')

       Lemmatizer  (token.lemma_):
         [14] unique_lemma_ratio  unique lemmas / content words  (vocab richness)

       NER  (doc.ents — statistical + EntityRuler):
         [15] clothing_ents     CLOTHING entity count
         [16] fit_ents          FIT entity count
         [17] material_ents     MATERIAL entity count
         [18] sentiment_pos     SENTIMENT_POS entity count
         [19] net_sentiment     SENTIMENT_POS − SENTIMENT_NEG

    Parameters
    ----------
    tfidf_max_features : int
        TF-IDF vocabulary size (GridSearchCV-tunable).
    tfidf_ngram_range : tuple
        N-gram range for TF-IDF.
    """

    _KEEP_POS = {'NOUN', 'VERB', 'ADJ', 'ADV'}

    def __init__(self, tfidf_max_features=500, tfidf_ngram_range=(1, 2)):
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range  = tfidf_ngram_range

    # ── internal helpers ──────────────────────────────────────────────────────

    def _combined_texts(self, X):
        return (X['Title'].fillna('').astype(str) + ' ' +
                X['Review Text'].fillna('').astype(str)).str.strip().tolist()

    def _spacy_pass(self, texts):
        """
        Single nlp.pipe() pass.  Returns:
          norm_texts : list[str]  — lemmatized, POS-filtered strings for TF-IDF
          feat_matrix: ndarray    — (n, 20) spaCy NLP feature matrix
        """
        norm_texts, feat_rows = [], []

        for doc in self.nlp_.pipe(texts, batch_size=256):
            alpha = [t for t in doc if t.is_alpha]
            wc    = len(alpha)

            # ── Normalised text for TF-IDF ────────────────────────────────
            tokens = [
                t.lemma_.lower()
                for t in alpha
                if not t.is_stop and len(t) > 2 and t.pos_ in self._KEEP_POS
            ]
            norm_texts.append(' '.join(tokens))

            # ── Numeric NLP features ──────────────────────────────────────
            # Tokenizer
            char_count      = len(doc.text)
            sent_count      = len(list(doc.sents))
            excl_count      = doc.text.count('!')
            quest_count     = doc.text.count('?')
            avg_word_length = sum(len(t) for t in alpha) / max(1, wc)

            # Statistical POS tagger
            adj_c  = sum(1 for t in alpha if t.pos_ == 'ADJ')
            adv_c  = sum(1 for t in alpha if t.pos_ == 'ADV')
            vrb_c  = sum(1 for t in alpha if t.pos_ == 'VERB')
            nou_c  = sum(1 for t in alpha if t.pos_ == 'NOUN')

            # Dependency parser — negation
            neg_c    = sum(1 for t in doc if t.dep_ == 'neg')
            neg_adj  = sum(
                1 for t in doc
                if t.pos_ == 'ADJ' and any(c.dep_ == 'neg' for c in t.children)
            )

            # Lemmatizer — vocabulary richness
            lemmas = [t.lemma_.lower() for t in alpha if not t.is_stop]
            ulr = len(set(lemmas)) / max(1, len(lemmas))

            # NER (statistical + EntityRuler)
            ec = {'CLOTHING':0,'FIT':0,'MATERIAL':0,
                  'SENTIMENT_POS':0,'SENTIMENT_NEG':0}
            for ent in doc.ents:
                if ent.label_ in ec:
                    ec[ent.label_] += 1

            feat_rows.append([
                char_count, wc, sent_count, excl_count, quest_count, avg_word_length,
                adj_c, adv_c, vrb_c, nou_c,
                adj_c / max(1, wc), adv_c / max(1, wc),
                neg_c, neg_adj, ulr,
                ec['CLOTHING'], ec['FIT'], ec['MATERIAL'],
                ec['SENTIMENT_POS'],
                ec['SENTIMENT_POS'] - ec['SENTIMENT_NEG'],
            ])

        return norm_texts, np.array(feat_rows)

    # ── sklearn API ───────────────────────────────────────────────────────────

    def fit(self, X, y=None):
        # Structured
        self.col_transformer_ = ColumnTransformer([
            ('num', StandardScaler(),                           NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore',
                                  sparse_output=False),        CATEGORICAL_FEATURES),
        ], remainder='drop')
        self.col_transformer_.fit(X)

        # spaCy single pass
        self.nlp_  = load_nlp()
        self.tfidf_ = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram_range,
            min_df=3,
            sublinear_tf=True,
        )
        norm_texts, _ = self._spacy_pass(self._combined_texts(X))
        self.tfidf_.fit(norm_texts)
        return self

    def transform(self, X):
        structured            = self.col_transformer_.transform(X)
        norm_texts, hc_feats  = self._spacy_pass(self._combined_texts(X))
        tfidf_feats           = self.tfidf_.transform(norm_texts).toarray()
        return np.hstack([structured, tfidf_feats, hc_feats])


# ── Feature name helper (used by notebook + dashboard) ───────────────────────

def get_feature_names(pipeline):
    """Return all feature names in the same order as the feature matrix columns."""
    fu   = pipeline.named_steps['features']
    num  = NUMERICAL_FEATURES
    cat  = list(fu.col_transformer_
                  .named_transformers_['cat']
                  .get_feature_names_out(CATEGORICAL_FEATURES))
    tfidf = fu.tfidf_.get_feature_names_out().tolist()
    nlp_feats = [
        'char_count','word_count','sent_count','excl_count','quest_count',
        'avg_word_length',
        'adj_count','adv_count','verb_count','noun_count',
        'adj_ratio','adv_ratio',
        'negation_count','neg_adj_count',
        'unique_lemma_ratio',
        'clothing_ents','fit_ents','material_ents',
        'sentiment_pos_ents','net_sentiment_ents',
    ]
    return num + cat + tfidf + nlp_feats
