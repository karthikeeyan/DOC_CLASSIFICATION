


RUN pip install numpy
RUN pip install pandas
RUN pip install sqlalchemy
RUN pip install pymysql
RUN pip install matplotlib
RUN pip install streamlit
RUN pip install doc2text


ENTRYPOINT ["streamlit", "run", "app.py"]
import pandas as pd
import numpy as np
import streamlit as st
import docx2txt
import pdfplumber
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from nltk.probability import FreqDist
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot  as plt
import plotly.express as px
pd.set_option('display.max_rows', 1000)

stop=set(stopwords.words('english'))


nltk.download('wordnet')
nltk.download('stopwords')
#nltk.download('omw-1.4')
import pickle
clf = pickle.load(open(r'C:\Users\karth\sweetha\clf_resume.pkl','rb'))
loaded_vec = pickle.load(open(r"C:\Users\karth\sweetha\tfidf_vect.pkl", "rb"))
resume = []

def display(doc_file):
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else :
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())
    return resume
    
def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)  





def mostcommon_words(cleaned,i):
    tokenizer = RegexpTokenizer(r'\w+')
    words=tokenizer.tokenize(cleaned)
    mostcommon=FreqDist(cleaned.split()).most_common(i)
    return mostcommon

def display_wordcloud(mostcommon):
    wordcloud=WordCloud(width=1000, height=600, background_color='black').generate(str(mostcommon))
    a=px.imshow(wordcloud)    
    st.plotly_chart(a)

def display_words(mostcommon_small):
    x,y=zip(*mostcommon_small)
    chart=pd.DataFrame({'keys': x,'values': y})
    fig=px.bar(chart,x=chart['keys'],y=chart['values'],height=700,width=700)
    st.plotly_chart(fig)


def display_result(result_pred):
    if result_pred==0:
        display='Peoplesoft Resume'
    elif result_pred==1:
        display='React JS Resume'
    elif result_pred==2:
        display='SQL Developer Lightning insight'
    elif result_pred==3:
        display ='workday Resume'
    return display

def main():
    st.title('DOCUMENT CLASSIFICATION')
    upload_file = st.file_uploader('Hey,Upload Your Resume ',
                                type= ['docx','pdf'],accept_multiple_files=True)
    if st.button("Process"):
        for doc_file in upload_file:
            if doc_file is not None:
                file_details = {'filename':[doc_file.name],
                               'filetype':doc_file.type.split('.')[-1].upper(),
                               'filesize':str(doc_file.size)+' KB'}
                file_type=pd.DataFrame(file_details)
                st.write(file_type.set_index('filename'))
                displayed=display(doc_file)
                cleaned=preprocess(display(doc_file))
                result_pred = clf.predict(loaded_vec.transform([cleaned]))
                
                string='The Uploaded file is belongs to  '+display_result(result_pred)
                st.header(string)
                
                st.subheader('WORDCLOUD')
                display_wordcloud(mostcommon_words(cleaned,100))
                
                st.header('Frequency of 20 Most Common Words :')
                display_words(mostcommon_words(cleaned,20))
                
                
            
    
    
    
if __name__ == '__main__':
    main()
