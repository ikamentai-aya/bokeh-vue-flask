from threading import Thread

from flask import Flask, render_template
from flask_cors import CORS
from tornado.ioloop import IOLoop

from bokeh.embed import server_document
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider,Div
from bokeh.plotting import figure
from bokeh.server.server import Server

from bokeh.plotting import figure, ColumnDataSource
from bokeh.models.widgets import Slider
from bokeh.models import HoverTool, Div, Scatter, Button, Spinner,RadioButtonGroup, TableColumn, DataTable
from bokeh.layouts import layout, Row, widgetbox,Column
import numpy as np
from sklearn.manifold import TSNE

from bokeh.palettes import Plasma256, mpl, magma, Category20
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from wordcloud import WordCloud

from module.tfidf import deleteShortText, TFIDFfilter
import collections
import pickle
import time

app = Flask(__name__)
CORS(app, resources={r'/*':{'origins':'*'}}, static_folder="static")

###データの読み込み
with open('static/paper-8813/report_content/content.pickle',mode='rb') as f:
    r_content = pickle.load(f)
    report_content = r_content[0]
    r_section_title = r_content[1]
    r_addition = r_content[2]
    
with open('static/paper-8813/video_content/slide_audio.pickle',mode='rb') as f:
    content = pickle.load(f)
    audio_content = content[0]

new_report_content, new_audio_content = deleteShortText(report_content,audio_content)

new_array, top_label_list,label, X, idf, vectorizer = TFIDFfilter(new_report_content+new_audio_content, 0.15)
new_array1, top_label_list1,label1, X1, idf1, vectorizer1 = TFIDFfilter(new_report_content+new_audio_content, 1.0)

new_report_vector = list(new_array[:len(new_report_content)])
new_audio_vector = list(new_array[len(new_report_content):])

print(len(new_report_content), len(new_audio_content))
def bkapp(doc):
    
    vector = np.array(new_report_vector+new_audio_vector)
    N1 = len(new_report_vector)
    belong = ['論文']*len(new_report_vector)+['音声']*len(new_audio_vector)
    default_color = ['green']*len(new_report_vector)+['blue']*len(new_audio_vector)
    index = list(range(len(new_report_vector)))+list(range(len(new_audio_vector)))
    
    #座標の準備
    tsne = TSNE(n_components=2, learning_rate='auto',init='random',perplexity=5)
    vector_embedd = tsne.fit_transform(vector)
    x, y = zip(*vector_embedd)
    x = list(x); y = list(y)
    
    #色の準備
    dbscan_dataset = cluster.DBSCAN(eps=1, min_samples=5, metric='euclidean').fit_predict(vector)
    color_num = max(len(set(dbscan_dataset))-1, 3)
    #color_palette = magma(color_num)
    color_palette = Category20[color_num]
    dbscan_color = [color_palette[i] if i >=0 else 'white' for i in dbscan_dataset]

    #最初にtSNEとDBCANをする
    source = ColumnDataSource(data={
        'x':x,
        'y':y,
        'index':index,
        'belong':belong,
        'default_color':default_color,
        'dbscan_color':dbscan_color,
        'content':new_report_content+new_audio_content,
        'group_text':['']*len(index),
        'top_word':top_label_list,
        'huti_color':Plasma256[:len(index)],
        'marker':['circle']*len(new_report_content)+['square']*len(new_audio_content)
    })
    line_x = [[x[i],x[i+1]] for i in range(len(index)-1)]
    line_y = [[y[i],y[i+1]] for i in range(len(index)-1)]
    
    line_source = ColumnDataSource(data={
        'x':line_x,
        'y':line_y,
        'color':['black']*len(line_x)
    })
    TOOLTIPS = [
        ('所属', '@belong'),
        ('インデックス', '@index'),
        ('top word', '@top_word')
    ]
    
    LABELS=[]
    ra_switch = RadioButtonGroup(labels=LABELS, active=0, height = 30)
    
    content = new_report_content+new_audio_content
    
    plot1 = figure(title='DBSCANの結果', height=400, width=400)
    plot1.multi_line(xs = 'x', ys='y', color="color", alpha=0.3, line_width=0.5, source=line_source)
    renderer = plot1.scatter(x='x', y='y', size=10, color='dbscan_color', alpha=0.5, source=source, line_color='huti_color', marker='marker')
    hover = HoverTool(tooltips=TOOLTIPS, renderers=[renderer])
    plot1.add_tools(hover)
    """
    plot2 = figure(title='元の色合い', height=400, width=400)
    plot2.scatter(x='x', y='y', size=10, color='huti_color', alpha=0.5, source=source, marker='marker')
    plot2.multi_line(xs = 'x', ys='y', color="color", alpha=0.3, line_width=0.5, source=line_source)
    #plot2.add_tools(HoverTool(tooltips=TOOLTIPS))
    """
    #tSNEのパラメーター
    perplexity_slider = Slider(title='perplexity【tSNE】', value=5, start=0, end=100, step=5)
    #DBSCANのパラメーター
    eps_slider = Slider(title='eps【DBSCAN】', value=1, start=0, end=1, step=0.01)
    min_samples_slider = Slider(title='min_samples【DBSCAN】', value=3, start=2, end=10, step=1)
    group_text_div = Div(text='',width=600, height=600)
    save_button = Button()
    wordcloud = Div(text='', height=300, width=400)
    global group_index
    group_index = {}
    
    def change_coordinate(attr, old, new):
        perplexity = perplexity_slider.value
        tsne = TSNE(n_components=2, learning_rate='auto',init='random',perplexity=perplexity)
        vector_embedd = tsne.fit_transform(vector)
        x, y = zip(*vector_embedd)
        x = list(x); y = list(y)
        source.data['x']=x
        source.data['y']=y
        line_x = [[x[i],x[i+1]] for i in range(len(index)-1)]
        line_y = [[y[i],y[i+1]] for i in range(len(index)-1)]
        line_source.data['x'] = line_x
        line_source.data['y'] = line_y
        
    def change_color(attr, old, new):
        eps = eps_slider.value
        min_samples = min_samples_slider.value
        dbscan_dataset = cluster.DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit_predict(vector)
        color_num = max(len(set(dbscan_dataset))-1, 3)
        color_palette = Category20[color_num]
        dbscan_color = [color_palette[i] if i >=0 else 'white' for i in dbscan_dataset]
        source.data['dbscan_color']=dbscan_color
        group = {}
        global group_index
        group_index = {}
        dbscan_dataset = dbscan_dataset.tolist()
        
        cluster_display=dict()
        cluster_words = dict()
        LABELS = []
        for i,g in enumerate(dbscan_dataset):
            if i >= N1:name = f'a{i-N1}'
            else: name = f'r{i}'
            
            if g !=-1 and g in cluster_display:
                group[g].append(name)
                group_index[g].append(i)
                cluster_display[g][i]='*';
                cluster_words[g] += list(top_label_list[i].split(' '))
            elif g !=-1:
                group[g]= [name]
                cluster_display[g] = ['_' for j in x]
                cluster_display[g][i]='*'
                group_index[g] = [i]
                cluster_words[g] = list(top_label_list[i].split(' '))
                LABELS.append(str(g))
        ra_switch.labels = LABELS
                
        group_text = ''''''
        for g in group:
            new_text = " ".join(group[g])
            display_text = ''.join(cluster_display[g])
            display_text = display_text[:N1]+'|'+display_text[N1:]
            count_word = collections.Counter(cluster_words[g])
            top_word_g = ''
            for word in dict(count_word):
                if count_word[word] > 2:top_word_g += f'【{word}】:{count_word[word]} '
            group_text += f'<p>【{g}】 {new_text}</p><p>{display_text}</p<p>{top_word_g}</p>'
            
        #print(group_text)
        group_text_div.text = group_text
        
        ###wordcloudの関数
        if 0 in group:
            text = ' '.join([source.data['content'][i] for i in group_index[0]])
            wc = WordCloud(width=400, height=300, background_color='white')
            wc.generate(text)
            wc.to_file('static/wc0.png')
            wordcloud.text = ''''''
            wordcloud.text ='''<img src="http://127.0.0.1:5000/static/wc0.png">'''
        else: wordcloud.text = ''''''
        
    def decideWordCloud(attr, old, new):
        global group_index
        cl_num = ra_switch.active
        text = ' '.join([source.data['content'][i] for i in group_index[cl_num]])
        wc = WordCloud(width=400, height=300, background_color='white')
        wc.generate(text)
        wc.to_file(f'static/wc{cl_num}.png')
        print('create wordcloud')
        wordcloud.text =f'''<img src="http://127.0.0.1:5000/static/wc{cl_num}.png">'''
            
        
        
    perplexity_slider.on_change('value',change_coordinate)
    
    eps_slider.on_change('value',change_color)
    min_samples_slider.on_change('value',change_color)
    ra_switch.on_change('active',decideWordCloud)
    
    doc.add_root(Column(Row(Column(plot1,ra_switch,wordcloud),Column(perplexity_slider, eps_slider, min_samples_slider,group_text_div))))
    #doc.add_root(Column(Row(plot1,plot2), perplexity_slider, eps_slider, min_samples_slider))
   
"""
def bkapp(doc):
  slider = Slider(start=1, end=10, value=1, step=1, title="Test")
  div = Div(text='',height=400,width=400)
  def change(attr,old,new):
    val = slider.value
    div.text=str(val)

  slider.on_change('value', change)

  doc.add_root(column(slider, div))
"""

@app.route('/', methods=['GET'])
def bkapp_page():
    script = server_document('http://localhost:5006/bkapp')
    return script

def bk_worker():
    # Can't pass num_procs > 1 in this configuration. If you need to run multiple
    # processes, see e.g. flask_gunicorn_embed.py
    server = Server({'/bkapp': bkapp}, io_loop=IOLoop(), allow_websocket_origin=["localhost:8081", "localhost:3000"])
    server.start()
    server.io_loop.start()

Thread(target=bk_worker).start()


if __name__=='__main__':
  app.run()


