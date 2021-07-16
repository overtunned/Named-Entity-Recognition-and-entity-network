import json
import logging

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import base64
# import tensorflow as tf
# import tensorflow_probability as tfp
from PIL import Image
import time
import codecs
import os
import spacy
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from afinn import Afinn
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
from spacy import displacy
from textblob import TextBlob

# def progress_bar(my,v):
#     my.progress(v)
from IPython.core.display import display, HTML

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


# Function for Sumy Summarization
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result
def tell_me_more():
    st.title('heres more')

    st.button('Back to NER')  # will change state and hence trigger rerun and hence reset should_tell_me_more

    

    st.button('Back to NER', key='back_again')  # will change state and hence trigger rerun and hence reset should_tell_me_more


def interactive_widgets():
    st.markdown('')
    # my_bar = st.progress(0)

    # st.sidebar.markdown("### Select a Model")
    # selected_model = st.sidebar.selectbox('', ['Novels', 'Medical', 'Articles'])    
    # selected_option = st.sidebar.radio('Radio', ["Type your text","Upload a text file"])
    # # progress_bar(my_bar,10)

    # if selected_option == "Type your text":
    #     st.subheader("Type Your Text")
    #     selected_txt = st.text_area("Enter Text","Type Here ..")
    #     # if st.button("Analyze"):
    #     #     nlp_result = text_analyzer(message)
    #     #     st.json(nlp_result)

	# # Type your text
    # if selected_option == "Upload a text file":
    #     selected_txt = st.sidebar.file_uploader('Click on Browse files to upload ')
    
    # st.sidebar.markdown('### Select the number of Top Entities ')
    # selected_topnum = st.sidebar.select_slider('Slide to select', options=[i for i in range(2,21)])

    # st.sidebar.markdown('### Select the threshhold of the intensity of the relationship ')
    # select_thresh = st.sidebar.select_slider('Slide to select', options=[0.00001,0.0001,0.0005, 0.001, 0.005, 0.01])
    # progress_bar(my_bar,50)
            # and sort by confidence, for now
#     galaxies = df
#     logging.info('Total galaxies: {}'.format(len(galaxies)))
#     valid = np.ones(len(df)).astype(bool)
#     for question, answers in questions.items():
#         answer, mean = current_selection.get(question, [None, None])  # mean is (min, max) limits
#         logging.info(f'Current: {question}, {answer}, {mean}')
#         if mean == None:  # happens when spiral count question not relevant
#             mean = (None, None)
#         if len(mean) == 1:
#             # streamlit sharing bug is giving only the higher value
#             logging.info('Streamlit bug is happening, working')
#             mean = (0., mean[0])
#         # st.markdown('{} {} {} {}'.format(question, answers, answer, mean))
#         if (answer is not None) and (mean is not None):
#             # this_answer = galaxies[question + '_' + answer + '_concentration_mean']
#             # all_answers = galaxies[[question + '_' + a + '_concentration_mean' for a in answers]].sum(axis=1)
#             this_answer = galaxies[question + '_' + answer + '_fraction']
#             all_answers = galaxies[[question + '_' + a + '_fraction' for a in answers]].sum(axis=1)
#             prob = this_answer / all_answers
#             within_limits = (np.min(mean) <= prob) & (prob <= np.max(mean))

#             preceding = True
#             if mean != (0., 1.):
#                 preceding = galaxies[question + '_proportion_volunteers_asked'] >= 0.5

#             logging.info('Fraction of galaxies within limits: {}'.format(within_limits.mean()))
#             valid = valid & within_limits & preceding

#     logging.info('Valid galaxies: {}'.format(valid.sum()))
#     st.markdown('{:,} of {:,} galaxies match your criteria.'.format(valid.sum(), len(valid)))

#     # selected = galaxies[valid].sample(np.min([valid.sum(), 16]))


#     # image_locs = [row['file_loc'].replace('/decals/png_native', '/galaxy_zoo/gz2') for _, row in selected.iterrows()]
#     # images = [np.array(Image.open(loc)).astype(np.uint8) for loc in image_locs]

#     # if show_posteriors is not 'none':
#     #     selected = galaxies[valid][:8]
#     #     question = show_posteriors
#     #     if question == 'spiral-count' or question == 'spiral-winding':
#     #         st.markdown('Sorry! You asked to see posteriors for "{}", but this demo app only supports visualing posteriors for questions with two answers. Please choose another option.'.format(question.capitalize().replace('-', ' ')))
#     #     else:
#     #         answers = questions[question]
#     #         selected_answer = current_selection[question][0]
#     #         for _, galaxy in selected.iterrows():
#     #             show_predictions(galaxy, question, answers, selected_answer)
#     # else:
#     # image_urls = ["https://panoptes-uploads.zooniverse.org/production/subject_location/02a32231-11c6-45b6-b448-fd85ec32fbd8.png"] * 16
#     selected = galaxies[valid][:40]
#     image_urls = selected['url']

#     opening_html = '<div style=display:flex;flex-wrap:wrap>'
#     closing_html = '</div>'
#     child_html = ['<img src="{}" style=margin:3px;width:200px;></img>'.format(url) for url in image_urls]

#     gallery_html = opening_html
#     for child in child_html:
#         gallery_html += child
#     gallery_html += closing_html

#     # st.markdown(gallery_html)
#     st.markdown(gallery_html, unsafe_allow_html=True)
#     # st.markdown('<img src="{}"></img>'.format(child_html), unsafe_allow_html=True)
#     # for image in images:
#     #     st.image(image, width=250)



    

# # def show_predictions(galaxy, question, answers, answer): 

# #     answer_index = answers.index(answer) 
# #     # st.markdown(answer_index)

# #     fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 3 * 1))

# #     total_votes = np.array(galaxy[question + '_total-votes']).astype(np.float32)  # TODO
# #     votes = np.linspace(0., total_votes)
# #     x = np.stack([votes, total_votes-votes], axis=-1)  # also need the counts for other answer, no
# #     votes_this_answer = x[:, answer_index]

# #     cycler = mpl.rcParams['axes.prop_cycle']
# #     # https://matplotlib.org/cycler/
# #     colors = [c['color'] for c in cycler]

# #     data =  [json.loads(galaxy[question + '_' + a + '_concentration']) for a in answers]
# #     all_samples = np.array(data).transpose(1, 0, 2)

# #     ax = ax0
# #     for model_n, samples in enumerate(all_samples):
# #         all_probs = []
# #         color = colors[model_n]
# #         n_samples = samples.shape[1]  # answer, dropout
# #         for d in range(n_samples):
# #             concentrations = tf.constant(samples[:, d].astype(np.float32))  # answer, dropout
# #             probs = tfp.distributions.DirichletMultinomial(total_votes, concentrations).prob(x)
# #             all_probs.append(probs)
# #             ax.plot(votes_this_answer, probs, alpha=.15, color=color)
# #         mean_probs = np.array(all_probs).mean(axis=0)
# #         ax.plot(votes_this_answer, mean_probs, linewidth=2., color=color)

# #     volunteer_response = galaxy[question + '_' + answer]
# #     ax.axvline(volunteer_response, color='k', linestyle='--')
        
# #     ax.set_xlabel(question.capitalize().replace('-', ' ') + ' "' + answer.capitalize().replace('-', ' ') + '" count')
# #     ax.set_ylabel(r'$p$(count)')

# #     ax = ax1
# #     # ax.imshow(np.array(Image.open(galaxy['file_loc'].replace('/media/walml/beta/decals/png_native', 'png'))))
# #     ax.imshow(np.array(Image.open(galaxy['file_loc'].replace('/media/walml/beta/decals/png_native', '/media/walml/beta1/galaxy_zoo/gz2'))))
# #     ax.axis('off')
        
# #     fig.tight_layout()
# #     st.write(fig)

st.set_page_config(
        layout="wide",
        page_title='GZ DECaLS',
        page_icon='gz_icon.jpeg'
    )


# @st.cache
# def load_data():
#     df_locs = ['decals_{}.csv'.format(n) for n in range(4)]
#     dfs = [pd.read_csv(df_loc) for df_loc in df_locs]
#     return pd.concat(dfs)
def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
    return allData

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

def entity_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    # st.write(HTML(displacy.render(docx, style="ent")))
    tokens = [ token.text for token in docx]
    entities = [(entity.text,entity.label_)for entity in docx.ents]
    allData = ['"Token":{},\n"Entities":{}'.format(tokens,entities)]
    return allData

def flatten(input_list):
    flat_list = []
    for i in input_list:
        if type(i) == list:
            flat_list += flatten(i)
        else:
            flat_list += [i]

    return flat_list

def common_words(path):
    with codecs.open(path) as f:
        words = f.read()
        words = json.loads(words)

    return set(words)

def calculate_align_rate(sentence_list):
    afinn = Afinn()
    sentiment_score = [afinn.score(x) for x in sentence_list]
    align_rate = np.sum(sentiment_score)/len(np.nonzero(sentiment_score)[0]) * -2

    return align_rate, sentiment_score

def name_entity_recognition(sentence):
    doc = nlp(sentence)
    # retrieve person and organization's name from the sentence
    name_entity = [x for x in doc.ents if x.label_ in ['PERSON', 'ORG']]
    # convert all names to lowercase and remove 's in names
    name_entity = [str(x).lower().replace("'s","") for x in name_entity]
    # split names into single words ('Harry Potter' -> ['Harry', 'Potter'])
    name_entity = [x.split(' ') for x in name_entity]
    # flatten the name list
    name_entity = flatten(name_entity)
    # remove name words that are less than 3 letters to raise recognition accuracy
    name_entity = [x for x in name_entity if len(x) >= 3]
    # remove name words that are in the set of 4000 common words
    name_entity = [x for x in name_entity if x not in words]

    return name_entity

def iterative_NER(sentence_list, threshold_rate=0.0005):
    output = []
    for i in sentence_list:
        name_list = name_entity_recognition(i)
        if name_list != []:
            output.append(name_list)
    output = flatten(output)
    from collections import Counter
    output = Counter(output)
    output = [x for x in output if output[x] >= threshold_rate * len(sentence_list)]

    return output

def top_names(name_list, novel, top_num=2):
    vect = CountVectorizer(vocabulary=name_list, stop_words='english')
    name_frequency = vect.fit_transform([novel.lower()])
    name_frequency = pd.DataFrame(name_frequency.toarray(), columns=vect.get_feature_names())
    name_frequency = name_frequency.T
    name_frequency = name_frequency.sort_values(by=0, ascending=False)
    name_frequency = name_frequency[0:top_num]
    names = list(name_frequency.index)
    name_frequency = list(name_frequency[0])

    return name_frequency, names

def calculate_matrix(name_list, sentence_list, align_rate):
    afinn = Afinn()
    sentiment_score = [afinn.score(x) for x in sentence_list]
    # calculate occurrence matrix and sentiment matrix among the top characters
    name_vect = CountVectorizer(vocabulary=name_list, binary=True)
    occurrence_each_sentence = name_vect.fit_transform(sentence_list).toarray()
    cooccurrence_matrix = np.dot(occurrence_each_sentence.T, occurrence_each_sentence)
    sentiment_matrix = np.dot(occurrence_each_sentence.T, (occurrence_each_sentence.T * sentiment_score).T)
    sentiment_matrix += align_rate * cooccurrence_matrix
    cooccurrence_matrix = np.tril(cooccurrence_matrix)
    sentiment_matrix = np.tril(sentiment_matrix)
    # diagonals of the matrices are set to be 0 (co-occurrence of name itself is meaningless)
    shape = cooccurrence_matrix.shape[0]
    cooccurrence_matrix[[range(shape)], [range(shape)]] = 0
    sentiment_matrix[[range(shape)], [range(shape)]] = 0

    return cooccurrence_matrix, sentiment_matrix

def matrix_to_edge_list(matrix, mode, name_list):
    edge_list = []
    shape = matrix.shape[0]
    lower_tri_loc = list(zip(*np.where(np.triu(np.ones([shape, shape])) == 0)))
    normalized_matrix = matrix / np.max(np.abs(matrix))
    if mode == 'co-occurrence':
        weight = np.log(2000 * normalized_matrix + 1) * 0.7
        color = np.log(2000 * normalized_matrix + 1)
    if mode == 'sentiment':
        weight = np.log(np.abs(1000 * normalized_matrix) + 1) * 0.7
        color = 2000 * normalized_matrix
    for i in lower_tri_loc:
        edge_list.append((name_list[i[0]], name_list[i[1]], {'weight': weight[i], 'color': color[i]}))

    return edge_list

def plot_graph(name_list, name_frequency, matrix, plt_name, mode, path=''):
    label = {i: i for i in name_list}
    edge_list = matrix_to_edge_list(matrix, mode, name_list)
    normalized_frequency = np.array(name_frequency) / np.max(name_frequency)

    plt.figure(figsize=(20, 20))
    G = nx.Graph()
    G.add_nodes_from(name_list)
    G.add_edges_from(edge_list)
    pos = nx.circular_layout(G)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    colors = [G[u][v]['color'] for u, v in edges]

    if mode == 'co-occurrence':
        nx.draw(G, pos, node_color='#A0CBE2', node_size=np.sqrt(normalized_frequency) * 4000, edge_cmap=plt.cm.Blues,
                linewidths=10, font_size=35, labels=label, edge_color=colors, with_labels=True, width=weights)
    elif mode == 'sentiment':
        nx.draw(G, pos, node_color='#A0CBE2', node_size=np.sqrt(normalized_frequency) * 4000,
                linewidths=10, font_size=35, labels=label, edge_color=colors, with_labels=True,
                width=weights, edge_vmin=-1000, edge_vmax=1000)
    else:
        raise ValueError("mode should be either 'co-occurrence' or 'sentiment'")

    plt.savefig(path + plt_name + '.png')

def ARI(novel, words, sentences):
    char  = len(novel)
    word= len(words)
    sent  = len(sentences)
    ARI = 4.71*(char/word) + 0.5*(word/sent) - 21.43
    return ARI


# def main(novel,options_selected):
    





if __name__ == '__main__':

    logging.basicConfig(level=logging.CRITICAL)

    st.title('Named Entity Relationships')
    st.subheader('by Abhi, jiss,sagar and tirumala sir')
    st.spinner()
    # st.balloons()
    should_tell_me_more = st.button('Tell me more')
    if should_tell_me_more:
        tell_me_more()
        st.markdown('---')
    else:
        st.markdown('---')
        interactive_widgets()

    st.sidebar.markdown("### Select the type")
    selected_model = st.sidebar.selectbox('', ['Novels'])#, 'Medical', 'Articles'])
    selected_option = st.sidebar.radio('Select', ["Type your text","Upload a text file"], index=0)
    # progress_bar(my_bar,10)

    if selected_option == "Type your text":
        st.subheader("Type Your Text")
        novel = st.text_area("","Type Here ...")
        # if st.button("Analyze"):
        #     nlp_result = text_analyzer(message)
        #     st.json(nlp_result)

	# Type your text
    if selected_option == "Upload a text file":
        selected_file = st.sidebar.file_uploader('Click on Browse files to upload ')
        if selected_file:
            novel = str(selected_file.read(),"utf-8").replace('\r', ' ').replace('\n', ' ').replace("\'", "'")

        
    if st.sidebar.button("Done"):    
    
        st.sidebar.markdown('### Select the number of Top Entities ')
        selected_topnum = st.sidebar.select_slider('Slide to select', options=[i for i in range(2,21)])

        st.sidebar.markdown('### Select the threshhold of the intensity of the relationship ')
        select_thresh = st.sidebar.select_slider('Slide to select', options=[0.00001,0.0001,0.0005, 0.001, 0.005, 0.01])

        nlp = spacy.load('en_core_web_sm')
        words = common_words('common_words.txt')



    if st.checkbox("Show Tokens and Lemma"):
        st.subheader("Tokenize Your Text")
        if novel:
            nlp_result = text_analyzer(novel)
            st.json(nlp_result)

    # Entity Extraction
    if st.checkbox("Show Named Entities"):
        st.subheader("Analyze Your Text")

        if novel:
            entity_result = entity_analyzer(novel)
            st.json(entity_result)

    # Sentiment Analysis
    if st.checkbox("Show Sentiment Analysis"):
        st.subheader("Analyse Your Text")
        if novel:
            blob = TextBlob(novel)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)

    # Summarization
    if st.checkbox("Show Text Summarization"):
        st.subheader("Summarize Your Text")
        if novel:
            st.text("Using Sumy Summarizer ..")
            summary_result = sumy_summarizer(novel)
        else:
            st.warning("Using Default Summarizer")
            st.text("Using Gensim Summarizer ..")
            summary_result = summarize(novel)


        st.success(summary_result)
    sentence_list = sent_tokenize(novel)
    word_list = word_tokenize(novel)
    uwords = set(word_list)

    if st.checkbox("Novel Details"):
        st.success("Total number of Characters :"+ str(len(novel)))
        st.success("Total number of words :"+ str(len(word_list)))
        st.success("Total unique words :"+ str(len(uwords)))
        st.success("Total number of sentecnes :"+ str(len(sentence_list)))
        

    if st.checkbox("ARI"):
        ari = ARI(novel,word_list,sentence_list)
        st.success("ARI : "+str(ari))

    if st.checkbox("Show Align Rate"):
        align_rate, sentiment_score = calculate_align_rate(sentence_list)
        sentence_with_score = pd.DataFrame([(w, t) for w,t in zip(sentence_list, sentiment_score)],columns=['sent','score'])
        st.success("Align Rate of the book :"+str(align_rate))

        if st.checkbox("Plot Sentiment Score distribution"):
            hist_values = np.histogram(sentence_with_score['score'])[0]
            st.bar_chart(hist_values)
            if st.checkbox("Show priliminary name list"):
                preliminary_name_list = iterative_NER(sentence_list, threshold_rate = select_thresh)
                if st.checkbox("Show top names frequency"):
                    name_frequency, name_list = top_names(preliminary_name_list, novel, selected_topnum)
                    cooccurrence_matrix, sentiment_matrix = calculate_matrix(name_list, sentence_list, align_rate)

                    if st.checkbox("Show cooccurance graph"):
                        options_selected['Show_cooccurance']=True

                    if st.checkbox("Show sentiment graph"):
                        options_selected['Show_senti']=True


    # main(novel, options_selected)
    
