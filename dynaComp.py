#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import numpy as np
import Backgammon
import flipped_agent
import torch
import time
from torch.autograd import Variable
device = torch.device('cpu')

# Define theta and theta_trans, implement load later
w1 = Variable(torch.randn(4*24,197, device = device, dtype=torch.float))
b1 = Variable(torch.zeros((4*24,1), device = device, dtype=torch.float))
w2 = Variable(torch.randn(2*24,4*24, device = device, dtype=torch.float))
b2 = Variable(torch.zeros((2*24,1), device = device, dtype=torch.float))
w3 = Variable(torch.randn(1,2*24, device = device, dtype=torch.float))
b3 = Variable(torch.zeros((1,1), device = device, dtype=torch.float))

# load permanent memo
load('dyna')


def forward(x):
	h = (torch.mm(w1,x) + b1).sigmoid()
	h2 = (torch.mm(w2,h) + b2).sigmoid()
	y = torch.mm(w3,h2) + b3

	return y.sigmoid()


def feature_encoding(board, doubleD):
	features = np.zeros(197) #24*8+4+1 positions*bips + kill + doubledice

	#Wet pips
	for i in range(1,25):
		features[0 + (i - 1)*4] = board[i] <= -1
		features[1 + (i - 1)*4] = board[i] <= -2
		features[2 + (i - 1)*4] = board[i] <= -3
		features[3 + (i - 1)*4] = board[i] <= -4

	# blek pips
	for i in range(25,49):
		features[0 + (i - 1)*4] = board[i-24] >= 1
		features[1 + (i - 1)*4] = board[i-24] >= 2
		features[2 + (i - 1)*4] = board[i-24] >= 3
		features[3 + (i - 1)*4] = board[i-24] >= 4

	features[192] = board[25] * 1/2
	features[193] = -board[26] * 1/2
	features[194] = board[27] * 1/15
	features[195] = -board[28] * 1/15

	features[196] = doubleD

	return features

def action(board, dice, player, i):

	doubleD = 1 if (dice[0]==dice[1] and i == 0) else 0
	if player == -1: board = flipped_agent.flip_board(np.copy(board))
	
	# check out the legal moves available for the throw
	possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player=1)

	# if there are no moves available
	if len(possible_moves) == 0:
			return []

	va = np.zeros(len(possible_boards))

	for i, possibleBoard in enumerate(possible_boards):
		x = Variable(torch.tensor(feature_encoding(possibleBoard, doubleD), dtype=torch.float, device = device)).view(197,1)
		va[i] = forward(x)

	move = possible_moves[np.argmax(va)]

	if player == -1: move = flipped_agent.flip_move(move)
	
	return move	

def load(name):
	try:
		w1 = torch.load(name)
		w2 = torch.load(name)
		w3 = torch.load(name)
		b1 = torch.load(name)
		b2 = torch.load(name)
		b3 = torch.load(name)
	except FileNotFoundError:
		print('starting fresh')

