class base_wordpro:
    # seperate list for punctuation
    plist = ['!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',
             ']','^','_','`','{','|','}','~']
    stop_w_eng = ['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','yours',
                'yourself','yourselves','he','him','his','himself','she',"she's",'her','hers','herself','it',"it's",'its','itself',
                'they','them','their','theirs','themselves','what','which','who','whom','this','that',"that'll",'these','those',
                'am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an',
                'the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between',
                'into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over',
                'under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few',
                'more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can',
                'will','just','don',"don't",'should',"should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",
                'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',
                "isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",'wasn',
                  "wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't"]
    # Add a function that alows customised stopwords by appending to the original list.
    def stop_w_selfadd(x):
        hw1_NLP.stop_w_eng.append(x)
    # Function that rest the stopword list
    def reset_stopword():
        hw1_NLP.stop_w_eng = ['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','yours',
                'yourself','yourselves','he','him','his','himself','she',"she's",'her','hers','herself','it',"it's",'its','itself',
                'they','them','their','theirs','themselves','what','which','who','whom','this','that',"that'll",'these','those',
                'am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an',
                'the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between',
                'into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over',
                'under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few',
                'more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can',
                'will','just','don',"don't",'should',"should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",
                'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',
                "isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",'wasn',
                  "wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't"]
    # 2.1 Function for lower case
    def lowerc(df):
        df.review = df.review.apply(lambda x:x.lower())
        return(df)
    # 2.2 Function for punctuation
    def removePunc(df):
        df.review = df.review.apply(lambda x: " ".join(x for x in x.split() if x not in hw1_NLP.plist))
        return(df)
    # 2.3 Function for stopwords
    def removestop(df):
        df.review = df.review.apply(lambda x: " ".join(x for x in x.split() if x not in hw1_NLP.stop_w_eng))
        return(df)
    # 2.4 function for tokenization
    def wordToken(df):
        df['wordToken'] = df.review.apply(lambda x: x.split())
        return(df)
    #----------------
    # From this point, the task will need nltk to complete
    # 2.5 Word stemming:
    def wordstem(df):
        stemming = PorterStemmer()
        df['review_stemmed'] = df.wordToken.apply(lambda x: [stemming.stem(word) for word in x])
        return(df)
    #2.6 Lemmatizing:
    def wordLemma(df):
        lemmatizer = WordNetLemmatizer()
        df['review_lemmatized'] = df.wordToken.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
        return(df)
    #2.7 Plot word frequency
    def wordFreqPlot(df,lab = 'All',top=10):
        if lab == 'neg':
            df = df[df.label==lab]
            outputT = 'Number of negative words:'
        elif lab =='pos':
            df = df[df.label==lab]
            outputT = 'Number of positive words:'
        else:
            df = df
            outputT = 'Number of words:'
        word = df.review_lemmatized.to_list()
        word = [item for sublist in word for item in sublist]
        word_counter = Counter(word)
        print(outputT,len(word_counter))
        TopN_common_words = word_counter.most_common()[:top]
        TopN_common_words = pd.DataFrame(TopN_common_words)
        TopN_common_words.columns = ['word', 'freq']
        TopN_common_words.sort_values(by='freq',ascending=True).plot(x='word', kind='barh')
        sorted_word_counts = sorted(list(word_counter.values()), reverse=True)
    #2.8 Remove frequent unimportant word
    def removeFreqUnimportant(df,ls,Lemma =True):
        if Lemma:
            df['review_lemmatized'] = df['review_lemmatized'].apply(lambda x: [y for y in x if y not in ls])
        else:
            df['review_stemmed'] = df['review_stemmed'].apply(lambda x: [y for y in x if y not in ls])
        return(df)