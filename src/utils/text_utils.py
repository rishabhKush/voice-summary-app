  import nltk
  from nltk.tokenize import sent_tokenize
  from nltk.corpus import stopwords

  nltk.download('punkt')
  nltk.download('stopwords')

  def split_into_sentences(text):
      """Splits text into sentences"""
      sentences = sent_tokenize(text)
      return sentences

  def get_stopwords():
      return set(stopwords.words('english'))
