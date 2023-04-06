import random
import json
import logging
import gensim
from gensim.models import Word2Vec
logging.basicConfig(level=logging.INFO,format="%(asctime)s-%(name)s-%(levelname)s-%(message)s")
domains={}
id2domain={}
id2url={}
url2id={}
url2label={}
url2domain={}
with open("formated_website_rank.txt","r") as fin:
    a=json.load(fin)
for key in a.keys():
    domain=a[key]
    domains[domain]={}
    domains[domain]['rank']=int(key)
    
with open("domain.csv","r") as fin:
    inputstr=fin.readline()
    inputstr=fin.readline()
    it=0
    while inputstr:
        inputlist=inputstr.split("\t")
        domain_id=int(inputlist[0])
        domain_name=inputlist[2]
        if domain_name not in domains:
            inputstr=fin.readline()
            continue
        if domains[domain_name]['rank']>=5000:
            inputstr=fin.readline()
            continue
        #logging.info(domain_name)
        domains[domain_name]["id"]=domain_id
        domains[domain_name]["url"]=[]
        id2domain[domain_id]=domain_name
        inputstr=fin.readline()
        it+=1      
        if it%1000==0:      
            logging.info("it:{}".format(it))
logging.info("step1 finished. Total domain num:"+str(it))

with open("domain_url.csv","r") as fin:
    #it=0
    inputstr=fin.readline()
    inputstr=fin.readline()
    while inputstr:
        inputlist=inputstr.split("\t")
        domain_id=int(inputlist[1])
        url_id=int(inputlist[2])
        if domain_id not in id2domain:
            inputstr=fin.readline()
            continue
        domain_name=id2domain[domain_id]
        if url_id not in domains[domain_name]["url"]:
            domains[domain_name]["url"].append(url_id)
        if url_id not in url2domain:
            url2domain[url_id]=[domain_id]
        else:
            url2domain[url_id].append(domain_id)
        inputstr=fin.readline()
        #it+=1
        #if it%100000==0:      
            #logging.info("it:{}".format(it))
logging.info("step2 finished. Total url num:{}".format(len(url2domain)))
it=0
track=0
with open("url.csv","r") as fin:
    inputstr=fin.readline()
    inputstr=fin.readline()
    while inputstr:
        inputlist=inputstr.split("\t")
        url_id=int(inputlist[0])
        if url_id not in url2domain:
            inputstr=fin.readline()
            continue
        url=inputlist[2]
        url_label=int(inputlist[14])
        if url_label==1:
            track+=1
        id2url[url_id]=url
        url2id[url]=url_id
        url2label[url_id]=url_label
        inputstr=fin.readline()
        it+=1
logging.info("step3 finishe.\ntotal url:{}\ntrack num:{}\n".format(it,track))

with open("id_url_label_new_5000.txt","w") as fout:
    for url in url2label:
        fout.write(str(url)+" "+id2url[url]+" "+str(url2label[url])+"\n")
   
length=20
dimen = 128
window = 5
walks=[]
time=10
def node2vec(walks):
    w2v_model= Word2Vec(walks, vector_size = dimen, window = window, min_count = 0, workers = 2, sg = 1, hs = 0, negative = 5)
    w2v_model.wv.save_word2vec_format("graph_embedding_new_5000.txt")

def random_walker(first_node,walk_length):
    #print(type(first_node))
    series = [first_node]
    
    for _ in range(1, walk_length):
        #print(type(first_node))
        if len(url2domain[series[-1]])>0:
            nodes_list = url2domain[series[-1]]
            domain_node = random.choice(nodes_list)
            domain_name=id2domain[domain_node]
            if len(domains[domain_name]["url"])>0:
                nodes_list = domains[domain_name]["url"]
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


