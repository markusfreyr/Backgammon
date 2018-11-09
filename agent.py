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

# define the parameters for the single hidden layer feed forward neural network
# randomly initialized weights with zeros for the biases
dtype = torch.cuda.FloatTensor

w1 = Variable(torch.randn((200,58), device = device, dtype=torch.float), requires_grad = True)
b1 = Variable(torch.zeros((200,1), device = device, dtype=torch.float), requires_grad = True)

w2 = Variable(torch.randn((1,200), device = device, dtype=torch.float), requires_grad = True)
b2 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)


Z_w1 = torch.zeros(w1.size(), device = device, dtype = torch.float)
Z_b1 = torch.zeros(b1.size(), device = device, dtype = torch.float)
Z_w2 = torch.zeros(w2.size(), device = device, dtype = torch.float)
Z_b2 = torch.zeros(b2.size(), device = device, dtype = torch.float)

w1_c = Variable(torch.randn((50,58), device = device, dtype=torch.float), requires_grad = True)
b1_c = Variable(torch.zeros((50,1), device = device, dtype=torch.float), requires_grad = True)

w2_c = Variable(torch.randn((1,50), device = device, dtype=torch.float), requires_grad = True)
b2_c = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)


Z_w1_c = torch.zeros(w1_c.size(), device = device, dtype = torch.float)
Z_b1_c = torch.zeros(b1_c.size(), device = device, dtype = torch.float)
Z_w2_c = torch.zeros(w2_c.size(), device = device, dtype = torch.float)
Z_b2_c = torch.zeros(b2_c.size(), device = device, dtype = torch.float)

w1_start = w1.detach().cpu().numpy()
w2_start = w2.detach().cpu().numpy()
print("w1")
print(w1)
print()
print("w2")
print(w2)

xold = []
count = 0
game_count = 0
ngames = 0


def action(board_copy,dice,player,i):
    global count, ngames, w1start, game_count
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
        x = Variable(torch.tensor(one_hot_encoding(board), dtype = torch.float, device = device)).view(58,1)
        #x = Variable(torch.tensor(one_hot_encoding(board), dtype = torch.float, device = device)).view(29,31)
        # now do a forward pass to evaluate the board's after-state value
        h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
        va[i] = y.sigmoid()

    count += 1
    game_count += 1



    if not Backgammon.game_over(possible_boards[np.argmax(va)]):
        update(possible_boards[np.argmax(va)])
    else:
        ngames += 1
        reward = 1 if player == 1 else -1
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

def update(board, reward=0,new_game=False):
    global Z_w1, Z_b1, Z_w2, Z_b2, xold, count, Z_w1_c, Z_b1_c, Z_w2_c, Z_b2_c, ngames, game_count
    alpha = 0.05 # step size for tabular learning
    alpha1 = 0.05 # step sizes using for the neural network (first layer)
    alpha2 = 0.05 # (second layer)
    alpha3 = 0.05 # (third layer)
    epsilon = 0.05 # exploration parameter used by both players
    lam_a = 0.1 # lambda parameter in TD(lam-bda)
    lam_c = 0.9
    gamma = 1 # for completeness

    if count > 1:
        x = Variable(torch.tensor(one_hot_encoding(board), dtype = torch.float, device = device)).view(58,1)
        # now do a forward pass to evaluate the board's after-state value
        h = torch.mm(w1_c,x) + b1_c # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        y = torch.mm(w2_c,h_sigmoid) + b2_c # multiply with the output weights w2 and add bias
        y_sigmoid = y.sigmoid()
        target_c = y_sigmoid.detach().cpu().numpy()

        h = torch.mm(w1_c,xold) + b1_c 
        h_sigmoid = h.sigmoid() 
        y_c = torch.mm(w2_c,h_sigmoid) + b2_c 
        y_c_sigmoid = y_c.sigmoid()
        delta_a = reward + gamma * target_c - y_c_sigmoid.detach().cpu().numpy() # delta fyrir að update-a actor-inn

        h = torch.mm(w1,x) + b1 
        h_sigmoid = h.sigmoid() 
        y = torch.mm(w2,h_sigmoid) + b2 
        y_sigmoid = y.sigmoid()
        target_a = y_sigmoid.detach().cpu().numpy()

        h = torch.mm(w1,xold) + b1 
        h_sigmoid = h.sigmoid() 
        y_a = torch.mm(w2,h_sigmoid) + b2 
        y_a_sigmoid = y_a.sigmoid()
        delta_c = reward + gamma * target_a - y_a_sigmoid.detach().cpu().numpy() # delta fyrir að updata-a critic-inn


        # using autograd and the contructed computational graph in pytorch compute all gradients
        y_a_sigmoid.backward()
        # update the eligibility traces using the gradients
        
        

        
        Z_w2 = gamma * lam_a * Z_w2 + w2.grad.data#/y_a_sigmoid
        Z_b2 = gamma * lam_a * Z_b2 + b2.grad.data#/y_a_sigmoid
        Z_w1 = gamma * 0.99 * Z_w1 + w1.grad.data#/y_a_sigmoid
        Z_b1 = gamma * lam_a * Z_b1 + b1.grad.data#/y_a_sigmoid
        

        # zero the gradients
        w2.grad.data.zero_()
        b2.grad.data.zero_()
        w1.grad.data.zero_()
        b1.grad.data.zero_()
        # perform now the update for the weights
        delta_a =  torch.tensor(delta_a, dtype = torch.float, device = device)
        w1.data = w1.data + alpha1 * delta_a * Z_w1
        b1.data = b1.data + alpha1 * delta_a * Z_b1
        w2.data = w2.data + alpha2 * delta_a * Z_w2
        b2.data = b2.data + alpha2 * delta_a * Z_b2


        # using autograd and the contructed computational graph in pytorch compute all gradients
        y_c_sigmoid.backward()
        # update the eligibility traces using the gradients
        Z_w2_c = gamma * lam_c * Z_w2_c + w2_c.grad.data
        Z_b2_c = gamma * lam_c * Z_b2_c + b2_c.grad.data
        Z_w1_c = gamma * lam_c * Z_w1_c + w1_c.grad.data
        Z_b1_c = gamma * lam_c * Z_b1_c + b1_c.grad.data
        # zero the gradients
        w2_c.grad.data.zero_()
        b2_c.grad.data.zero_()
        w1_c.grad.data.zero_()
        b1_c.grad.data.zero_()
        # perform now the update for the weights
        delta_c =  torch.tensor(delta_c, dtype = torch.float, device = device)
        w1_c.data = w1_c.data + alpha1 * delta_c * Z_w1_c
        b1_c.data = b1_c.data + alpha1 * delta_c * Z_b1_c
        w2_c.data = w2_c.data + alpha2 * delta_c * Z_w2_c
        b2_c.data = b2_c.data + alpha2 * delta_c * Z_b2_c
        if count % 10000 == 0:
            print()
            print("w1")
            print(count)
            print(w1_c)
            print("Z_w1")
            print(Z_w1_c.detach().cpu().numpy())
            print("w2")
            print(count)
            print(w2_c)
            print("Z_w2")
            print(Z_w2_c.detach().cpu().numpy())
            print("b1")
            print(count)
            print(torch.transpose(b1_c,0,1))
            print("Z_b1")
            print(torch.transpose(Z_b1_c,0,1).detach().cpu().numpy())
            print("b2")
            print(count)
            print(torch.transpose(b2_c,0,1))
            print("Z_b2")
            print(torch.transpose(Z_b2_c,0,1).detach().cpu().numpy())
            print("y_a")
            print(y_a)
            print(y_a_sigmoid)
            print("y_c")
            print(y_c)
            print(y_c_sigmoid)
            
    if new_game:
        game_count = 0
    xold = Variable(torch.tensor(one_hot_encoding(board), dtype = torch.float, device = device)).view(58,1)



def one_hot_encoding(data, nb_classes=31):
    one_hot = np.zeros(2*len(data))

    for i in range(0, len(data)):
        if (data[i] > 0):
            one_hot[i] = data[i] / 15
        else:
            one_hot[i+29] = data[i] / 15

    return one_hot
"""

def one_hot_encoding(data, nb_classes=31):
    "Convert an iterable of indices to one-hot encoded labels."
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets.astype(int)]
"""

