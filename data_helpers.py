#encoding=utf-8
def load_data_and_labels(file):
    """-------适合不同类型的文件（word_self_entity, father_entity）"""
    sentences=[]
    label=[]
    with open(file,"r",encoding="utf-8") as fr:
        for line in fr.readlines():
            line=line.strip().split("\t")
            # print(line)
            if len(line)==2:
                sentences.append(line[1])
                if line[0]=="0":
                    label.append([0,1])
                if line[0]=="1":
                    label.append([1,0])
            else:
                sentences.append("<UNK>")
                if line[0] == "0":
                    label.append([0, 1])
                if line[0] == "1":
                    label.append([1, 0])

    return sentences,label



