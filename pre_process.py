import pandas as pd
import random
import gensim
from gensim.models import Word2Vec
#data=pd.read_csv("domain_url.csv")
id2url={}
url2id={}
url2label={}
domain2url={}
url2domain={}
id2domain={}
domain2id={}
track=0
total=0
print("start")
it=0
with open("url.csv","r") as fin:
    inputstr=fin.readline()
    inputstr=fin.readline()
    #it=0
    while inputstr and it<1000000:
        inputlist=inputstr.split("\t")
        #print(inputlist)
        total+=1
        if int(inputlist[14])==1:
            track+=1
        id2url[int(inputlist[0])]=inputlist[2]
        url2id[inputlist[2]]=int(inputlist[0])
        url2label[int(inputlist[0])]=int(inputlist[14])
        inputstr=fin.readline()
        it+=1
print("total url:{}\ntrack num:{}\n".format(total,track))
print("step1 finished.")
with open("id_url_label.txt","w") as fout:
    for url in url2label:
        fout.write(str(url)+" "+id2url[url]+" "+str(url2label[url])+"\n")
        

with open("domain_url.csv","r") as fin:
    #it=0
    inputstr=fin.readline()
    inputstr=fin.readline()
    while inputstr:
        inputlist=inputstr.split("\t")
        domain_id=int(inputlist[1])
        url_id=int(inputlist[2])
        if url_id not in id2url:
            inputstr=fin.readline()
            continue
        if url_id not in url2domain:
            url2domain[url_id]=[domain_id]
        else:
            url2domain[url_id].append(domain_id)
        if domain_id not in domain2url:
            domain2url[domain_id]=[url_id]
        else:
            domain2url[domain_id].append(url_id)
        inputstr=fin.readline()
        #it+=1
print("step2 finished.")
rmurl=[]
for url in id2url:
    if url not in url2domain:
        rmurl.append(url)
for url in rmurl:
    del id2url[url]
    if url2label[url]==1:
        track-=1
    total-=1
print("total url:{}\ntrack num:{}\n".format(total,track))

with open("domain.csv","r") as fin:
    inputstr=fin.readline()
    inputstr=fin.readline()
    #it=0
    while inputstr:
        inputlist=inputstr.split("\t")
        domain_id=int(inputlist[0])
        if domain_id not in domain2url:
            inputstr=fin.readline()
            continue
        #print(inputlist)
        id2domain[int(inputlist[0])]=inputlist[2]
        domain2id[inputlist[2]]=int(inputlist[0])
        inputstr=fin.readline()
        #it+=1        
print("step3 finished.")


length=50
dimen = 128
window = 5
walks=[]
time=10
def node2vec(walks):
    w2v_model= Word2Vec(walks, vector_size = dimen, window = window, min_count = 0, workers = 2, sg = 1, hs = 0, negative = 5)
    w2v_model.wv.save_word2vec_format("graph_embedding.txt")

def random_walker(first_node,walk_length):
    #print(type(first_node))
    series = [first_node]
    
    for _ in range(1, walk_length):
        #print(type(first_node))
        if len(url2domain[series[-1]])>0:
            nodes_list = url2domain[series[-1]]
            domain_node = random.choice(nodes_list)
            if len(domain2url[domain_node])>0:
                nodes_list = domain2url[domain_node]
                url_node=random.choice(nodes_list)
                series.append(url_node)
        else:
            break
    return series
def deep_walker(time,length):
    node_series=[]
    for i in range(time):
        print("round"+str(i))
        for node in id2url:
            node_series.append(random_walker(node,length))
    return node_series
walks=deep_walker(time,length)
node2vec(walks)


