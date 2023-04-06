import logging
from transformers import BertTokenizer,BertModel,BertForMaskedLM
logging.basicConfig(level=logging.INFO,format="%(asctime)s-%(name)s-%(levelname)s-%(message)s")

tokenizer=BertTokenizer.from_pretrained("bert-base-cased")
bert=BertModel.from_pretrained("bert-base-cased")
fout=open("bert_embedding_new_5000.txt","w")
with open("id_url_label_new2.txt","r") as fin:
    inputstr=fin.readline()
    it=0
    while inputstr:
        inputlist=inputstr.strip().split(" ")
        id_url=int(inputlist[0])
        url=inputlist[1]
        tokens=tokenizer.encode_plus(text=url,return_tensors='pt')
        try:
            model_out=bert(**tokens)
        except:
            logging.info(id_url)
            logging.info(url)
            #print(model_out)
        else:
            fout.write(str(id_url)+" "+str(model_out["last_hidden_state"][0][0].detach().numpy().tolist())+"\n")
        inputstr=fin.readline()
        it+=1
        if it%1000==0:
            logging.info(it)
fout.close()
#tokens=tokenizer.tokenize(text)
#print(tokens)
#input_ids=tokenizer.convert_tokens_to_ids(tokens)
#output=BertModel
#print(model_out["last_hidden_state"])
