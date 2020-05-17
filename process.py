# This function is used for loading the data and converting the same into a structured datastucture, other than splitting here I am removing the special charecters specially.
def load_data_and_labels_one(filename,encoding='utf-8'):
    words=[]
    pos=[]    
    sent=[]
    label1=[]
    tempz="""(),.<>?$#@"!%&*:;'~`^=-_+\|{}[]/"""
    removal=[i for i in tempz]
    with open(filename, encoding=encoding) as f:
        for line in f:
            line = line.rstrip()
            if line:
                word, pos_,parser_,ner_ = line.split()
                if word!="-DOCSTART-" and word not in removal and pos_ not in removal:
                    words.append(word)
                    pos.append(pos_)
            else:
                if words!=[] and pos!=[]:
                    sent.append(words)
                    label1.append(pos)
                    words,pos= [], []

    return sent, label1
# This function also works same like the previous one but here I am not removing the special charecters.
def load_data_and_labels_two(filename,encoding='utf-8'):
    words=[]
    parser=[]
    ner=[]    
    sent=[]
    label2=[]
    label3=[]
    tempz="""(),.<>?$#@"!%&*:;'~`^=-_+\|{}[]/"""
    removal=[i for i in tempz]
    with open(filename, encoding=encoding) as f:
        for line in f:
            line = line.rstrip()
            if line:
                word, pos_,parser_,ner_ = line.split()
                if word!="-DOCSTART-" and parser_ not in removal and ner_ not in removal:
                    words.append(word)
                    parser.append(parser_)
                    ner.append(ner_)
            else:
                if words!=[] and ner!=[] and parser!=[]:
                    sent.append(words)
                    label2.append(parser)
                    label3.append(ner)
                    words, parser, ner = [], [], []

    return sent, label2, label3
