#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import numpy as np
import Backgammon
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import flipped_agent
device = torch.device('cpu')

# Þetta gæti verið partur af modelinu bara, það er nice
xold1 = []
xold2 = []
count = 0
win = 0
double = 0

#### NEW 
class Net(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in,H1) #in 29X31 - 31X29*2 out 29 X 29*2
        self.linear2 = nn.Linear(H1, H2) #in 29X29*2 - 29*2X1 out 29 X 1
        self.linear3 = nn.Linear(H2, D_out) #in (29X1)T - 29 X 1 out 1x1 smá mix útaf transpose..

        self.Z_w1_p1 = torch.transpose(torch.zeros((D_in,H1), device = device, dtype = torch.float), 0, 1) # size 31x58
        self.Z_b1_p1 = torch.zeros(H1, device = device, dtype = torch.float) # size 1
        self.Z_w2_p1 = torch.zeros(H2,H1, device = device, dtype = torch.float) # size 1x58
        self.Z_b2_p1 = torch.zeros(H2, device = device, dtype = torch.float) # size 1
        self.Z_w3_p1 = torch.zeros(D_out,H2, device = device, dtype = torch.float) # size 1x29
        self.Z_b3_p1 = torch.zeros(D_out, device = device, dtype = torch.float) # size 1

        self.Z_w1_p2 = torch.transpose(torch.zeros((D_in,H1), device = device, dtype = torch.float), 0, 1) # size 31x58
        self.Z_b1_p2 = torch.zeros(H1, device = device, dtype = torch.float) # size 1
        self.Z_w2_p2 = torch.zeros(H2,H1, device = device, dtype = torch.float) # size 1x58
        self.Z_b2_p2 = torch.zeros(H2, device = device, dtype = torch.float) # size 1
        self.Z_w3_p2 = torch.zeros(D_out,H2, device = device, dtype = torch.float) # size 1x29
        self.Z_b3_p2 = torch.zeros(D_out, device = device, dtype = torch.float) # size 1

    def forward(self, x):
        h1 = self.linear1(x).tanh()
        h2 = self.linear2(h1).tanh()
        y = self.linear3(h2).tanh()

        return y


#Global actor
actorModel = Net((31*29)+1, 29*2, 20, 1)
criticModel = Net((31*29)+1, 29*2, 20, 1)


try:
    a = "actorModel_A001_L09_ele.txt"
    c = "criticModel_A001_L09_ele.txt"

    actorData = torch.load(a)
    actorModel.load_state_dict(actorData)
    actorModel.eval()

    criticData = torch.load(c)
    criticModel.load_state_dict(criticData)
    criticModel.eval()
    print('loading: ',a,c)
except FileNotFoundError:
    print('starting fresh')


def action(board_copy,dice,player,i):
    global count, win, double
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy

    double = 1 if (dice[0]==dice[1] and i == 0) else 0

    if player == -1: board_copy = flipped_agent.flip_board(np.copy(board_copy))
    
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player=1)

    # if there are no moves available
    if len(possible_moves) == 0:
        return [], win
    
    # make the best move according to the policy

    na = len(possible_moves)
    va = np.zeros(na)
    for t in range(0, na):
        move = possible_moves[t]

        # this does not change the board_copy variable
        board = np.copy(board_copy)
        for m in move:
            board = Backgammon.update_board(board, m, 1)

        # encode the board to create the input
        x = Variable(torch.tensor(one_hot_encoding(board), dtype = torch.float, device = device))
        #print(x.size(), 'x')
        y = actorModel.forward(x)
        va[t] = y

    if i == 0: count += 1

    # Get the best move, update board with that move
    # this changes the board_copy variable
    move = possible_moves[np.argmax(va)]

    if player == -1: move = flipped_agent.flip_move(move)
    if player == -1: board_copy = flipped_agent.flip_board(board_copy)
    
    """ looser_board = np.copy(board_copy)
    for m in move:
        board_copy = Backgammon.update_board(board_copy, m, player)

    if not Backgammon.game_over(board_copy) and not Backgammon.check_for_error(board_copy):
        update(board_copy, player)
    else:
        win += 1
        update(board_copy, player, reward=1)
        update(board_copy, player*-1, reward=-1) """
                

    
    return move, win


def update(board, player, reward=0):
    alpha = 0.005 # step size for tabular learning
    alpha1 = 0.005 # step sizes using for the neural network (first layer)
    alpha2 = 0.005 # (second layer)
    alpha3 = 0.005 # (third layer)
    epsilon = 0.005 # exploration parameter used by both players
    lam_a = 0.7 # lambda parameter in TD(lam-bda)
    lam_c = 0.7
    gamma = 1 # for completeness
    global Z_w1, Z_b1, Z_w2, Z_b2,Z_w3, Z_b3, xold1, xold2,count


    if count > 2:
        x = Variable(torch.tensor(one_hot_encoding(board), dtype = torch.float, device = device))
        y = criticModel.forward(x)
        target_c = y.detach().cpu().numpy()

        if player == 1:
            y_c = criticModel.forward(xold1)
        else:
            y_c = criticModel.forward(xold2)
        delta_a = reward + gamma * target_c - y_c.detach().cpu().numpy() # Skipta þessu út fyrir policy gradient?

        y = actorModel.forward(x)
        target_a = y.detach().cpu().numpy()

        if player == 1:
            y_a = actorModel.forward(xold1)
        else:
            y_a = actorModel.forward(xold2)

        delta_c = reward + gamma * target_a - y_a.detach().cpu().numpy()
        # using autograd and the contructed computational graph in pytorch compute all gradients
        log_y_a = torch.log(y_a)
        log_y_a.backward() 
                
        param = list(actorModel.parameters())
        w1 = param[0]
        b1 = param[1]
        w2 = param[2]
        b2 = param[3]
        w3 = param[4]
        b3 = param[5]

        if player == 1:
            # update the eligibility traces using the gradients
            actorModel.Z_w3_p1 = gamma * lam_a * actorModel.Z_w3_p1 + w3.grad.data 
            actorModel.Z_b3_p1 = gamma * lam_a * actorModel.Z_b3_p1 + b3.grad.data 
            actorModel.Z_w2_p1 = gamma * lam_a * actorModel.Z_w2_p1 + w2.grad.data
            actorModel.Z_b2_p1 = gamma * lam_a * actorModel.Z_b2_p1 + b2.grad.data
            actorModel.Z_w1_p1 = gamma * lam_a * actorModel.Z_w1_p1 + w1.grad.data
            actorModel.Z_b1_p1 = gamma * lam_a * actorModel.Z_b1_p1 + b1.grad.data
            # zero the gradients
            w3.grad.data.zero_()
            b3.grad.data.zero_()
            w2.grad.data.zero_()
            b2.grad.data.zero_()
            w1.grad.data.zero_()
            b1.grad.data.zero_()
            # perform now the update for the weights
            delta_a =  torch.tensor(delta_a, dtype = torch.float, device = device)
            # delta2 er 1x1 en b1,b2,b3 eru bara 1, margföldum því með delta[0] fyrir bias
            w1.data = w1.data + alpha1 * delta_a * actorModel.Z_w1_p1
            b1.data = b1.data + alpha1 * delta_a[0] * actorModel.Z_b1_p1
            w2.data = w2.data + alpha2 * delta_a * actorModel.Z_w2_p1
            b2.data = b2.data + alpha2 * delta_a[0] * actorModel.Z_b2_p1
            w3.data = w3.data + alpha3 * delta_a * actorModel.Z_w3_p1
            b3.data = b3.data + alpha3 * delta_a[0] * actorModel.Z_b3_p1
            
            y_c.backward()
            
            param = list(criticModel.parameters())
            w1 = param[0]
            b1 = param[1]
            w2 = param[2]
            b2 = param[3]
            w3 = param[4]
            b3 = param[5]
            
            
            # update the eligibility traces using the gradients
            criticModel.Z_w3_p1 = gamma * lam_c * criticModel.Z_w3_p1 + w3.grad.data 
            criticModel.Z_b3_p1 = gamma * lam_c * criticModel.Z_b3_p1 + b3.grad.data 
            criticModel.Z_w2_p1 = gamma * lam_c * criticModel.Z_w2_p1 + w2.grad.data
            criticModel.Z_b2_p1 = gamma * lam_c * criticModel.Z_b2_p1 + b2.grad.data
            criticModel.Z_w1_p1 = gamma * lam_c * criticModel.Z_w1_p1 + w1.grad.data
            criticModel.Z_b1_p1 = gamma * lam_c * criticModel.Z_b1_p1 + b1.grad.data
            # zero the gradients
            w3.grad.data.zero_()
            b3.grad.data.zero_()
            w2.grad.data.zero_()
            b2.grad.data.zero_()
            w1.grad.data.zero_()
            b1.grad.data.zero_()
            # perform now the update for the weights
            delta_c =  torch.tensor(delta_c, dtype = torch.float, device = device)
            # delta2 er 1x1 en b1,b2,b3 eru bara 1, margföldum því með delta[0] fyrir bias
            w1.data = w1.data + alpha1 * delta_a * criticModel.Z_w1_p1
            b1.data = b1.data + alpha1 * delta_a[0] * criticModel.Z_b1_p1
            w2.data = w2.data + alpha2 * delta_a * criticModel.Z_w2_p1
            b2.data = b2.data + alpha2 * delta_a[0] * criticModel.Z_b2_p1
            w3.data = w3.data + alpha3 * delta_a * criticModel.Z_w3_p1
            b3.data = b3.data + alpha3 * delta_a[0] * criticModel.Z_b3_p1
        else:
                        # update the eligibility traces using the gradients
            actorModel.Z_w3_p2 = gamma * lam_a * actorModel.Z_w3_p2 + w3.grad.data 
            actorModel.Z_b3_p2 = gamma * lam_a * actorModel.Z_b3_p2 + b3.grad.data 
            actorModel.Z_w2_p2 = gamma * lam_a * actorModel.Z_w2_p2 + w2.grad.data
            actorModel.Z_b2_p2 = gamma * lam_a * actorModel.Z_b2_p2 + b2.grad.data
            actorModel.Z_w1_p2 = gamma * lam_a * actorModel.Z_w1_p2 + w1.grad.data
            actorModel.Z_b1_p2 = gamma * lam_a * actorModel.Z_b1_p2 + b1.grad.data
            # zero the gradients
            w3.grad.data.zero_()
            b3.grad.data.zero_()
            w2.grad.data.zero_()
            b2.grad.data.zero_()
            w1.grad.data.zero_()
            b1.grad.data.zero_()
            # perform now the update for the weights
            delta_a =  torch.tensor(delta_a, dtype = torch.float, device = device)
            # delta2 er 1x1 en b1,b2,b3 eru bara 1, margföldum því með delta[0] fyrir bias
            w1.data = w1.data + alpha1 * delta_a * actorModel.Z_w1_p2
            b1.data = b1.data + alpha1 * delta_a[0] * actorModel.Z_b1_p2
            w2.data = w2.data + alpha2 * delta_a * actorModel.Z_w2_p2
            b2.data = b2.data + alpha2 * delta_a[0] * actorModel.Z_b2_p2
            w3.data = w3.data + alpha3 * delta_a * actorModel.Z_w3_p2
            b3.data = b3.data + alpha3 * delta_a[0] * actorModel.Z_b3_p2
            
            y_c.backward()
            
            param = list(criticModel.parameters())
            w1 = param[0]
            b1 = param[1]
            w2 = param[2]
            b2 = param[3]
            w3 = param[4]
            b3 = param[5]
            
            
            # update the eligibility traces using the gradients
            criticModel.Z_w3_p2 = gamma * lam_c * criticModel.Z_w3_p2 + w3.grad.data 
            criticModel.Z_b3_p2 = gamma * lam_c * criticModel.Z_b3_p2 + b3.grad.data 
            criticModel.Z_w2_p2 = gamma * lam_c * criticModel.Z_w2_p2 + w2.grad.data
            criticModel.Z_b2_p2 = gamma * lam_c * criticModel.Z_b2_p2 + b2.grad.data
            criticModel.Z_w1_p2 = gamma * lam_c * criticModel.Z_w1_p2 + w1.grad.data
            criticModel.Z_b1_p2 = gamma * lam_c * criticModel.Z_b1_p2 + b1.grad.data
            # zero the gradients
            w3.grad.data.zero_()
            b3.grad.data.zero_()
            w2.grad.data.zero_()
            b2.grad.data.zero_()
            w1.grad.data.zero_()
            b1.grad.data.zero_()
            # perform now the update for the weights
            delta_c =  torch.tensor(delta_c, dtype = torch.float, device = device)
            # delta2 er 1x1 en b1,b2,b3 eru bara 1, margföldum því með delta[0] fyrir bias
            w1.data = w1.data + alpha1 * delta_a * criticModel.Z_w1_p2
            b1.data = b1.data + alpha1 * delta_a[0] * criticModel.Z_b1_p2
            w2.data = w2.data + alpha2 * delta_a * criticModel.Z_w2_p2
            b2.data = b2.data + alpha2 * delta_a[0] * criticModel.Z_b2_p2
            w3.data = w3.data + alpha3 * delta_a * criticModel.Z_w3_p2
            b3.data = b3.data + alpha3 * delta_a[0] * criticModel.Z_b3_p2
                        



    if player == 1:
        xold1 = Variable(torch.tensor(one_hot_encoding(board), dtype = torch.float, device = device))
    else:
        xold2 = Variable(torch.tensor(one_hot_encoding(board), dtype = torch.float, device = device))

    if reward == -1:
        xold1 = []
        xold2 = []
        count = 0



def one_hot_encoding(data, nb_classes=31):
    global double
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    oneHot = np.eye(nb_classes)[targets.astype(int)].flatten()
    return np.append(oneHot, [double])

def save():
    torch.save(actorModel.state_dict(), 'ActorModel_minus')
    torch.save(criticModel.state_dict(), 'CriticModel_minus')
    







def main():
    model = Net(31, 29*2, 1, 29)
    board = np.zeros(29)
    board[1] = -2
    board[12] = -5
    board[17] = -3
    board[19] = -5
    board[6] = 5
    board[8] = 3
    board[13] = 5
    board[24] = 2

    x = Variable(torch.tensor(one_hot_encoding(board), dtype = torch.float, device = device))
    y = model.forward(x);

    ##Ath mymodel file-inn hann inniheldur gögnin
    torch.save(model.state_dict(), "mymodel")
    try:
        test = torch.load("ekkitil")
    except FileNotFoundError:
        print('starting fresh')

    """ print(model.state_dict()) 
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    print(test1) """

    ## test1 er það sama og var save-að svo þetta er nice
    ## Þetta mundi búa til nýtt model fyrir gömul gögn (Y)
    ## newModel = ActorNet(31, 29*2, 1, 29)
  ## newModel.load_state_dict(test1)
  ## newModel.eval()



if __name__ == '__main__':
    main()
