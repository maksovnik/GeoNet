

f = open("ntable.txt","r")


def ret(line):
    line = [i for i in line.strip().split(" ") if i!='']
    snd = line[-4:]
    fst = line[:-4]
    return [' '.join(fst)] + snd

def classification_report_to_latex(lines):
    lines = [i for i in lines if i != '\n']
    title = lines[0].strip()
    columns = ["state"] + [i for i in lines[1].strip().split(" ") if i!='']
    print(' & '.join(columns)+"\\\\")
    dataLines = [] 
    for i in range(2,len(lines)-3):
        b = ret(lines[i])
        dataLines.append(b)
    dataLines = sorted(dataLines, reverse=True,key=lambda x: x[3])
    

    for i in dataLines:
        print(' & '.join(i)+"\\\\")

    a = [i for i in lines[-3].strip().split(" ") if i!='']
    a = [a[0],"","",a[1],a[2]]
    print(' & '.join(a)+"\\\\")
    q = ret(lines[-2])
    w = ret(lines[-1])
    print(' & '.join(q)+"\\\\")
    print(' & '.join(w))
    


classification_report_to_latex(f.readlines())