    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import utils.text_utils as text_utils
    import config

    class TextSummarizer:
        def __init__(self, top_n = config.SUMMARY_TOP_SENTENCES):
            self.top_n = top_n
        def summarize(self, sentences):
            """Summarizes text using TF-IDF"""
            stop_words = text_utils.get_stopwords()
            vectorizer = TfidfVectorizer(stop_words=stop_words)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            top_sentence_indices = np.argsort(sentence_scores)[::-1][:self.top_n]
            selected_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
            return selected_sentences
