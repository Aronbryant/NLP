#-*-encoding:utf-8-*-
class HMM(object):
    def __init__(self) -> None:
        self.status = ['B','I','E','S']#词位状态
        self.A = {}#状态转移概率
        self.B = {}#观测概率
        self.PI = {}#初始概率

    def train(self,path):
        stateCounter = {}#统计状态出现次数
        def initParameters():
            for state in self.status:
                self.A[state] = {s:0.0 for s in self.status}
                self.PI[state] = 0.0
                self.B[state] = {}
                stateCounter[state] = 0
        def get_Label(text):
            out = []
            if(len(text) == 1):
                out.append('S')
            else:
                out += ['B'] + ['I']*(len(Text) -2)+['E']
            return out
        initParameters()
        lineCounter = 0
        with open(path,encoding='utf-8')  as f:
            for line in f.readlines():
                lineCounter +=1
                line = line.strip()
                if not line:
                    continue
                word_list = [i for i in line if i != ' ']
                linelist = line.split()
                line_state = []
                for w in linelist:
                    line_state.extend(get_Label(w))

                try:
                     assert len(word_list) == len(line_state)
                except:
                    continue
                for k,v in enumerate(line_state):
                    stateCounter[v] +=1
                    if k==0:
                        self.PI +=1
                    else:
                        self.A[line_state[k-1]][v] +=1
                        self.B[line_state[k]][word_list[k]] = \
                        self.B[line_state[k]].get(word_list[k],0) + 1.0

        self.PI ={k:v*1.0 / lineCounter for k,v in self.PI.items()}
        self.A = {k:{k1:v1 / stateCounter[k] for k1,v1 in v.items()} \
            for k,v in self.A.items()}
        self.B = {k:{k1:v1+1 / stateCounter[k] for k1,v1 in v.items()} \
            for k,v in self.B.items()}
    def cut(self,text):
        #加载模型进行分词
        prob,pos_list = self.viterbi(text,self.status,self.PI,self.A,self.B)
        begin,next = 0, 0
        for i,char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin: i+1]
                next = i+1
            elif pos == 'S':
                yield char
                next = i+1
        if next < len(text):
            yield text[next:]
        def viterbi(self,text,states,startP,transP,emitP):
        V = [{}]
        path = {}
        for state in states:
            V[0][state] = startP[state] * emitP[state].get(text[0],0)
            path[state] = [state]

        for t in range(1,len(text)):
            V.append({})
            newpath = {}
            neverseen = text[t] not in emitP['S'].keys() and \
                text[t] not in emitP['I'].keys() and \
                    text[t] not in emitP['E'].keys() and \
                        text[t] not in emitP['B'].keys() 
            for y in states:
                emitP_ = emitP[y].get(text[t],0) if not neverseen else 1.0
                (prob,state) = max([(V[t-1][y0] * transP[y0].get(y,0)*emitP_,
                y0) for y0 in states if V[t-1][y0] >0])
                V[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath
        if emitP['I'].get(text[-1],0) > emitP['S'].get(text[-1],0):
            (prob,state) = max([(V[len(text) - 1][y],y) for y in ('E','I')])
        else:
            (prob,state) = max([(V[len(text) - 1][y],y) for y in states])

        return (prob,path[state])