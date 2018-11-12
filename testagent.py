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
won_games = 0

#### NEW 
class Net(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Net, self).__init__()
        """
        self.linear1 = nn.Linear(D_in,H1) #in 29X31 - 31X29*2 out 29 X 29*2
        self.linear2 = nn.Linear(H1, H2) #in 29X29*2 - 29*2X1 out 29 X 1
        self.linear3 = nn.Linear(D_out, H2) #in (29X1)T - 29 X 1 out 1x1 smá mix útaf transpose..

        self.Z_w1_p1 = torch.transpose(torch.zeros((D_in,H1), device = device, dtype = torch.float), 0, 1) # size 31x58
        self.Z_b1_p1 = torch.zeros(H1, device = device, dtype = torch.float) # size 1
        self.Z_w2_p1 = torch.zeros(H2,H1, device = device, dtype = torch.float) # size 1x58
        self.Z_b2_p1 = torch.zeros(H2, device = device, dtype = torch.float) # size 1
        self.Z_w3_p1 = torch.zeros(H2,D_out, device = device, dtype = torch.float) # size 1x29
        self.Z_b3_p1 = torch.zeros(H2, device = device, dtype = torch.float) # size 1

        self.Z_w1_p2 = torch.transpose(torch.zeros((D_in,H1), device = device, dtype = torch.float), 0, 1) # size 31x58
        self.Z_b1_p2 = torch.zeros(H1, device = device, dtype = torch.float) # size 1
        self.Z_w2_p2 = torch.zeros(H2,H1, device = device, dtype = torch.float) # size 1x58
        self.Z_b2_p2 = torch.zeros(H2, device = device, dtype = torch.float) # size 1
        self.Z_w3_p2 = torch.zeros(H2,D_out, device = device, dtype = torch.float) # size 1x29
        self.Z_b3_p2 = torch.zeros(H2, device = device, dtype = torch.float) # size 1
        """
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
        self.Z_w3_p2 = torch.zeros(H2,D_out, device = device, dtype = torch.float) # size 1x29
        self.Z_b3_p2 = torch.zeros(H2, device = device, dtype = torch.float) # size 1

    def forward(self, x):
        h1 = self.linear1(x).sigmoid()
        h2 = self.linear2(h1).sigmoid()
        y = self.linear3(h2).sigmoid()
        #h1 = F.relu(self.linear1(x)).sigmoid()
        #h2 = F.relu(self.linear2(h1)).sigmoid()
        #y = F.relu(self.linear3(torch.transpose(h2, 0, 1))).sigmoid()
        return y


#Global actor
#actorModel = Net(31, 29*2, 1, 29)
#criticModel = Net(31, 29*2, 1, 29)

actorModel = Net(31*29, 29*2, 20, 1)
criticModel = Net(31*29, 29*2, 20, 1)

try:
    actorData = torch.load("ActorModel")
    actorModel.load_state_dict(actorData)
    actorModel.eval()

    criticData = torch.load("CriticModel")
    criticModel.load_state_dict(criticData)
    criticModel.eval()
except FileNotFoundError:
    print('starting fresh')


def action(board_copy,dice,player,i):
    global count, won_games
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy

    if player == -1: board_copy = flipped_agent.flip_board(np.copy(board_copy))
    
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player=1)

    # if there are no moves available
    if len(possible_moves) == 0:
        return []
    
    # make the best move according to the policy

    na = len(possible_moves)
    va = np.zeros(na)
    for i in range(0, na):
        #print(i,"i")
        move = possible_moves[i]
        board = possible_boards[i]


        # encode the board to create the input
        x = Variable(torch.tensor(one_hot_encoding(board), dtype = torch.float, device = device))

        y = actorModel.forward(x)
        
        va[i] = y

    count += 1
    
    if (count - 1) % 1000 == 0:
        print(va)
        print(list(actorModel.parameters())[4])
    '''
    if np.abs(board_copy[-1]) > 10 or np.abs(board_copy[-2]) > 10:
        print(board_copy,"before state")
        print(possible_boards[np.argmax(va)],"after state")
        print("checking error from agent")
        print(Backgammon.check_for_error(possible_boards[np.argmax(va)]),"check for error")
        print(possible_moves[np.argmax(va)])
        print(dice)
        print()
        print(possible_boards)
        print()
    '''

    if not Backgammon.game_over(possible_boards[np.argmax(va)]):
        update(possible_boards[np.argmax(va)], player)
    else:
        update(possible_boards[np.argmax(va)],player, reward=1)
        won_games += 1
        #print(won_games,"won_games")
        #update(flipped_agent.flip_board(possible_boards[np.argmax(va)]), player, reward=-1) 
    	

    move = possible_moves[np.argmax(va)]
    if player == -1: move = flipped_agent.flip_move(move)
    return move

def update(board, player, reward=0):
    alpha = 0.05 # step size for tabular learning
    alpha1 = 0.05 # step sizes using for the neural network (first layer)
    alpha2 = 0.05 # (second layer)
    alpha3 = 0.05 # (third layer)
    epsilon = 0.05 # exploration parameter used by both players
    lam_a = 0.9 # lambda parameter in TD(lam-bda)
    lam_c = 0.9
    gamma = 1 # for completeness
    global Z_w1, Z_b1, Z_w2, Z_b2,Z_w3, Z_b3, xold1, xold2,count,won_games


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
            actorModel.Z_w1_p2 = gamma * lam_a * actorModel.Z_w1_p2+ w1.grad.data
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



def one_hot_encoding(data, nb_classes=31):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets.astype(int)].flatten()
    #return np.eye(nb_classes)[targets.astype(int)]

def save():
	torch.save(actorModel.state_dict(), 'ActorModel')
	torch.save(criticModel.state_dict(), 'CriticModel')
	







def main():
	model = ActorNet(31, 29*2, 1, 29)
	board = np.zeros(29)
	board[1] = -2
	board[12] = -5
	board[17] = -3
	board[19] = -5
	board[6] = 5
	board[8] = 3
	board[13] = 5
	board[24] = 2

	x = Variable(torch.tensor(one_hot_encoding(board), dtype = torch.float, device = device)).view(29,31)
	y = model.forward(x)

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
