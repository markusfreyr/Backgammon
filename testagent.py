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
from torch.autograd import Variable
device = torch.device('cpu')

# ef Z_.. breyturnar eru partur af modelinu má taka þetta út
w1 = Variable(torch.randn((31*2,29), device = device, dtype=torch.float), requires_grad = True)
b1 = Variable(torch.zeros((31*2,31), device = device, dtype=torch.float), requires_grad = True)
w2 = Variable(torch.randn((1,31*2), device = device, dtype=torch.float), requires_grad = True)
b2 = Variable(torch.zeros((1,31), device = device, dtype=torch.float), requires_grad = True)
w3 = Variable(torch.randn((31,1), device = device, dtype=torch.float), requires_grad = True)
b3 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)

# Þetta gæti verið partur af modelinu bara, það er nice
Z_w1 = torch.zeros(w1.size(), device = device, dtype = torch.float)
Z_b1 = torch.zeros(b1.size(), device = device, dtype = torch.float)
Z_w2 = torch.zeros(w2.size(), device = device, dtype = torch.float)
Z_b2 = torch.zeros(b2.size(), device = device, dtype = torch.float)
Z_w3 = torch.zeros(w3.size(), device = device, dtype = torch.float)
Z_b3 = torch.zeros(b3.size(), device = device, dtype = torch.float)

xold = []
count = 0

#### NEW 
class ActorNet(torch.nn.Module):
	def __init__(self, D_in, H1, H2, D_out):
		super(ActorNet, self).__init__()
		self.linear1 = torch.nn.Linear(D_in,H1) #in 29x31 31x29*2 out 29x29*2
		self.linear2 = torch.nn.Linear(H1, H2) #in 29x29*2 29*2x1 out 29x1
		self.linear3 = torch.nn.Linear(D_out, H2) #in 1x29 29x1 out 1x1 smá mix útaf transpose..

	def forward(self, x):
		h1 = self.linear1(x).sigmoid()
		h2 = self.linear2(h1).sigmoid()
		y = self.linear3(torch.transpose(h2, 0, 1)).sigmoid()

		return y


#Global actor
actorModel = ActorNet(31, 29*2, 1, 29)

def action(board_copy,dice,player,i):
    global count
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player)

    # if there are no moves available
    if len(possible_moves) == 0:
        return []
    
    # make the best move according to the policy

    na = len(possible_moves)
    va = np.zeros(na)
    for i in range(0, na):
        move = possible_moves[i]
        board = possible_boards[i]

        # encode the board to create the input
        x = Variable(torch.tensor(one_hot_encoding(board), dtype = torch.float, device = device)).view(29,31)
        y = actorModel.forward(x)
        va[i] = y

    count += 1

    if not Backgammon.game_over(possible_boards[np.argmax(va)]):
        update(possible_boards[np.argmax(va)])
    else:
        reward = 1 if player == 1 else 0
        update(possible_boards[np.argmax(va)],reward)

    return possible_moves[np.argmax(va)]
    

    # policy missing, returns a random move for the time being
    #
    #
    #
    #
    #
"""     move = possible_moves[np.random.randint(len(possible_moves))]

    return move """

def update(board, reward=0):
    alpha = 0.05 # step size for tabular learning
    alpha1 = 0.05 # step sizes using for the neural network (first layer)
    alpha2 = 0.05 # (second layer)
    alpha3 = 0.05 # (third layer)
    epsilon = 0.05 # exploration parameter used by both players
    lam = 0.4 # lambda parameter in TD(lam-bda)
    gamma = 1 # for completeness
    global Z_w1, Z_b1, Z_w2, Z_b2,Z_w3, Z_b3, xold, count

    if count > 1:
        x = Variable(torch.tensor(one_hot_encoding(board), dtype = torch.float, device = device)).view(29,31)
        y = actorModel.forward(x)
        target = y.detach().cpu().numpy()


        y = actorModel.forward(xold)
        delta2 = reward + gamma * target - y.detach().cpu().numpy() # Skipta þessu út fyrir policy gradient?
        # using autograd and the contructed computational graph in pytorch compute all gradients
        y.backward() 
				
        param = list(model.parameters())
        w1 = param[0].grad.data
        b1 = param[1].grad.data
        w2 = param[2].grad.data
        b2 = param[3].grad.data
        w3 = param[4].grad.data
        b3 = param[5].grad.data

        # update the eligibility traces using the gradients
        Z_w3 = gamma * lam * Z_w3 + w3.grad.data 
        Z_b3 = gamma * lam * Z_b3 + b3.grad.data 
        Z_w2 = gamma * lam * Z_w2 + w2.grad.data
        Z_b2 = gamma * lam * Z_b2 + b2.grad.data
        Z_w1 = gamma * lam * Z_w1 + w1.grad.data
        Z_b1 = gamma * lam * Z_b1 + b1.grad.data
        # zero the gradients
        w3.grad.data.zero_()
        b3.grad.data.zero_()
        w2.grad.data.zero_()
        b2.grad.data.zero_()
        w1.grad.data.zero_()
        b1.grad.data.zero_()
        # perform now the update for the weights
        delta2 =  torch.tensor(delta2, dtype = torch.float, device = device)
        w1.data = w1.data + alpha1 * delta2 * Z_w1
        b1.data = b1.data + alpha1 * delta2 * Z_b1
        w2.data = w2.data + alpha2 * delta2 * Z_w2
        b2.data = b2.data + alpha2 * delta2 * Z_b2
        w3.data = w3.data + alpha3 * delta2 * Z_w3
        b3.data = b3.data + alpha3 * delta2 * Z_b3


    xold = Variable(torch.tensor(one_hot_encoding(board), dtype = torch.float, device = device)).view(29,31)



def one_hot_encoding(data, nb_classes=31):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets.astype(int)]







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
	y = model.forward(x);

	##Ath mymodel file-inn hann inniheldur gögnin
	torch.save(model.state_dict(), "mymodel")

	test1 = torch.load("mymodel")

	print(model.state_dict()) 
	print('\n')
	print('\n')
	print('\n')
	print('\n')
	print(test1)

	## test1 er það sama og var save-að svo þetta er nice
	## Þetta mundi búa til nýtt model fyrir gömul gögn (Y)
	## newModel = ActorNet(31, 29*2, 1, 29)
  ## newModel.load_state_dict(test1)
  ## newModel.eval()



if __name__ == '__main__':
	main()
