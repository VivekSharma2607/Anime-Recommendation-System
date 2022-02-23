import flask
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

app = flask.Flask(__name__ , template_folder = 'templates')

anime = pd.read_csv('Dataset/anime.csv')

ge_str = anime['genre'].str.split(',').astype(str)
tfidf = TfidfVectorizer(stop_words='english' , ngram_range=(1,4) , min_df = 0)
tfidf_matrix = tfidf.fit_transform(ge_str)
tfidf_matrix.shape

sig = sigmoid_kernel(tfidf_matrix , tfidf_matrix)
idx = pd.Series(anime.index , index = anime['name']).drop_duplicates()
all_title = [anime['name'][i] for i in range(len(anime['name']))]
def predict(title , sig = sig):
    indexes = idx[title]
    sig_scores = list(enumerate(sig[indexes]))
    sig_scores = sorted(sig_scores , key = lambda x : x[1] , reverse = True)
    sig_scores = sig_scores[1:11]
    anime_idx = [i[0] for i in sig_scores]
    return pd.DataFrame({'Anime Name' : anime['name'].iloc[anime_idx].values,
                         'Rating' : anime['rating'].iloc[anime_idx].values})

#mess = input("Enter anime name = ")
#predict(mess)

@app.route('/' , methods=['GET' , 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))

    if flask.request.method == 'POST':
        a_name = flask.request.form['anime_name']
        a_name = a_name.title()

        if a_name not in all_title:
            return(flask.render_template('no.html' , name = a_name))
        else:
            res_final = predict(a_name)
            names = []
            ratings = []
            for i in range(len(res_final)):
                names.append(res_final.iloc[i][0])
                ratings.append(res_final.iloc[i][1])
            return(flask.render_template('answer.html' , anime_name = names , ratings = ratings , search_name = a_name))
            
            

if __name__ == '__main__':
    app.run()
