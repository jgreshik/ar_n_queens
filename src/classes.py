class Square():
    def __init__(self, id_, popu_=False):
        self.id=id_
        self.popu=popu_
    def set_id(self,id_):
        self.id=id_
    def set_popu(self,popu_):
        self.popu=popu_

class Queen():
    def __init__(self, loc_, safe_=True):
        self.loc=int(loc_)
        self.safe=safe_
    def set_loc(self,loc_):
        self.loc=loc_
    def set_safe(self,safe_):
        self.safe=safe_

class Board ():
    def __init__(self):
        self.width=8
        self.height=8
        self.squares=[Square(id_=i) for i in range(self.width*self.height)]
        self.queens=[]
    def update_safe(self):
        for q in range(len(self.queens)):
            danger=0
            # check row
            row=int(self.queens[q].loc/self.height)
            col=int(self.queens[q].loc%self.width)
            for i in range(self.height): danger+=self.squares[row*self.height+i].popu*1             
            # check col
            for i in range(self.width): danger+=self.squares[col+self.width*i].popu*1             
            # check ascending diag
            # below queen
            for i in range(1,min(col+1,self.height-row)):
                danger+=self.squares[self.queens[q].loc+self.width*i-i].popu*1
            # above queen
            for i in range(1,min(self.width-col,row+1)):
                danger+=self.squares[self.queens[q].loc-self.width*i+i].popu*1
            # check descending diag
            # below queen
            for i in range(1,min(self.height-row,self.width-col)):
                danger+=self.squares[self.queens[q].loc+self.width*i+i].popu*1
            # above queen
            for i in range(1,min(col+1,row+1)):
                danger+=self.squares[self.queens[q].loc-self.width*i-i].popu*1
            danger-=2
            if danger>0: self.queens[q].safe=False
