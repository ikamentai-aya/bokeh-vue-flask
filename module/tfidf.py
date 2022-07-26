from sklearn.feature_extraction.text import TfidfVectorizer

def deleteShortText(report_content,audio_content):
  new_report_content = []
  new_audio_content = []

  for content in report_content:
      if len(content.split()) > 12:
          new_report_content.append(content)
  for content in audio_content:
      if len(content.split()) > 12:
          new_audio_content.append(content)

  return new_report_content, new_audio_content

def TFIDFfilter(corpus, max_df):
    ##TFIDFの計算
    vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, max_df=max_df)
    vectorizer.fit(corpus)
    #print(vectorizer.idf_)
    X = vectorizer.transform(corpus)
    X = X.toarray()
    
    #TFIDFのラベル,idfを獲得
    label = vectorizer.get_feature_names_out()
    idf = vectorizer.idf_
    
    #TFIDFのトップ20単語のTFを計算
    new_array = []
    vector_n = len(X[0])
    top_label_list = []
    
    for x_index, x in enumerate(X):
        n_array = [0 for i in x]
        sort_x, sort_index = zip(*sorted(zip(x, list(range(vector_n)))))
        
        count = 0
        sort_i = 0
        reverse_sort_index = sort_index[::-1]
        while count <= 20:
            i = reverse_sort_index[sort_i]
            #if idf[i] > 3: n_array[i] = corpus[x_index].count(label[i])
            n_array[i] = corpus[x_index].count(label[i])
            count += 1; sort_i += 1
        new_array.append(n_array)
        top_label = [label[i] for i in sort_index[::-1][:10]] 
        top_label_list.append(' '.join(top_label))
        
    return new_array, top_label_list, label, X, idf,vectorizer