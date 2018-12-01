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
from collections import defaultdict
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

w1_trans = []
b1_trans = []
w2_trans = []
b2_trans = []
w3_trans = []
b3_trans = []

def forward(x):
	h = (torch.mm(w1,x) + b1).sigmoid()
	h2 = (torch.mm(w2,h) + b2).sigmoid()
	y = torch.mm(w3,h2) + b3

	h_t = (torch.mm(w1_trans,x) + b1_trans).sigmoid()
	h2_t = (torch.mm(w2_trans,h_t) + b2_trans).sigmoid()
	y_t = torch.mm(w3_trans,h2_t) + b3_trans
	return y.sigmoid() + y_t.sigmoid()


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


def search(board,player):
	global w1_trans,b1_trans,w2_trans,b2_trans,w3_trans,b3_trans
	startState = np.copy(board)
	xold1 = []
	xold2 = []
	start = time.time()
	time_available = 1

	if Backgammon.game_over(board):
		return
	t=0
	#while time_available - (time.time() - start) > 0:
	while t <= 30 and time_available - (time.time() - start) > 0:
		t = t+1
		
		# Eligibility
		z1_w1 = Variable(torch.zeros((4*24,197), device = device, dtype=torch.float))
		z1_b1 = Variable(torch.zeros((4*24,1), device = device, dtype=torch.float))
		z1_w2 = Variable(torch.zeros((2*24,4*24), device = device, dtype=torch.float))
		z1_b2 = Variable(torch.zeros((2*24,1), device = device, dtype=torch.float))
		z1_w3 = Variable(torch.zeros((1,2*24), device = device, dtype=torch.float))
		z1_b3 = Variable(torch.zeros((1,1), device = device, dtype=torch.float))

		z2_w1 = Variable(torch.zeros((4*24,197), device = device, dtype=torch.float))
		z2_b1 = Variable(torch.zeros((4*24,1), device = device, dtype=torch.float))
		z2_w2 = Variable(torch.zeros((2*24,4*24), device = device, dtype=torch.float))
		z2_b2 = Variable(torch.zeros((2*24,1), device = device, dtype=torch.float))
		z2_w3 = Variable(torch.zeros((1,2*24), device = device, dtype=torch.float))
		z2_b3 = Variable(torch.zeros((1,1), device = device, dtype=torch.float))

		dice = Backgammon.roll_dice()
		a = action(np.copy(startState), dice, player, 0)

		count = 0
		first = True;
		while not Backgammon.game_over(board) and not Backgammon.check_for_error(board):
			for i in range(1+int(dice[0] == dice[1])):
				count += 1 if i != 1 else count + 0
				if first and dice[0] == dice[1]: i + 1
				first = False

				#Execute action
				new_board = np.copy(board) # new board is old board if no move
				for m in a:
					new_board = Backgammon.update_board(new_board, m, player)

				reward = 1 if Backgammon.game_over(new_board) else 0

				switch = 1 if i == 0 and dice[0] == dice[1] else -1
				
				#action for next player or same if double roll
				doubleD = 1 if switch == 1 else 0
				dice = Backgammon.roll_dice()
				new_action = action(np.copy(new_board), dice, player*switch,i)
				
				if count > 3:
					if player == 1:
						x = Variable(torch.tensor(feature_encoding(board, doubleD), dtype=torch.float, device = device)).view(197,1)
						y = forward(x)
						yold = forward(xold1)
						delta = reward + y - yold # TD-error
						
						# update theta
						thetaUpdate_trans(z1_w1,z1_b1,z1_w2,z1_b2,z1_w3,z1_b3, delta)
						#elligibillyboy
						z1_w1,z1_b1,z1_w2,z1_b2,z1_w3,z1_b3 = eligibillyUpdate(z1_w1,z1_b1,z1_w2,z1_b2,z1_w3,z1_b3, yold)

					else:
						# Flip?
						flipped = flipped_agent.flip_board(np.copy(board))
						x = Variable(torch.tensor(feature_encoding(flipped, doubleD), dtype=torch.float, device = device)).view(197,1)
						y = forward(x)
						yold = forward(xold2) # xold already flipped
						delta = reward + y - yold # TD-error
						
						# update theta
						thetaUpdate_trans(z2_w1,z2_b1,z2_w2,z2_b2,z2_w3,z2_b3, delta)
						#elligibillyboy
						z2_w1,z2_b1,z2_w2,z2_b2,z2_w3,z2_b3 = eligibillyUpdate(z2_w1,z2_b1,z2_w2,z2_b2,z2_w3,z2_b3, yold)

				if player == 1:
					xold1 = Variable(torch.tensor(feature_encoding(board, doubleD), dtype=torch.float, device = device)).view(197,1)
				else:
					#Flip?
					flipped = flipped_agent.flip_board(np.copy(board))
					xold2 = Variable(torch.tensor(feature_encoding(flipped, doubleD), dtype=torch.float, device = device)).view(197,1)

				board = new_board
				a = new_action

				player = player * switch
			
			

def learn(n):
	#global theta1, theta_trans1, theta2, theta_trans2
	global w1,b1,w2,b2,w3,b3, w1_trans,b1_trans,w2_trans,b2_trans,w3_trans,b3_trans

	#A = defaultdict(list)
	#B = defaultdict(list)
	#theta = 0 gert annarsstaðar
	#while True:
	for t in range(n):
		print(t)
		if (t+1)%500 == 0: save('./dynaW/')
		count = 0
		xold1 = []
		xold2 = []
		# Eligibility
		z1_w1 = Variable(torch.zeros((4*24,197), device = device, dtype=torch.float))
		z1_b1 = Variable(torch.zeros((4*24,1), device = device, dtype=torch.float))
		z1_w2 = Variable(torch.zeros((2*24,4*24), device = device, dtype=torch.float))
		z1_b2 = Variable(torch.zeros((2*24,1), device = device, dtype=torch.float))
		z1_w3 = Variable(torch.zeros((1,2*24), device = device, dtype=torch.float))
		z1_b3 = Variable(torch.zeros((1,1), device = device, dtype=torch.float))

		z2_w1 = Variable(torch.zeros((4*24,197), device = device, dtype=torch.float))
		z2_b1 = Variable(torch.zeros((4*24,1), device = device, dtype=torch.float))
		z2_w2 = Variable(torch.zeros((2*24,4*24), device = device, dtype=torch.float))
		z2_b2 = Variable(torch.zeros((2*24,1), device = device, dtype=torch.float))
		z2_w3 = Variable(torch.zeros((1,2*24), device = device, dtype=torch.float))
		z2_b3 = Variable(torch.zeros((1,1), device = device, dtype=torch.float))

		# initialize transient
		w1_trans = Variable(torch.randn(4*24,197, device = device, dtype=torch.float))
		b1_trans = Variable(torch.zeros((4*24,1), device = device, dtype=torch.float))
		w2_trans = Variable(torch.randn(2*24,4*24, device = device, dtype=torch.float))
		b2_trans = Variable(torch.zeros((2*24,1), device = device, dtype=torch.float))
		w3_trans = Variable(torch.randn(1,2*24, device = device, dtype=torch.float))
		b3_trans = Variable(torch.zeros((1,1), device = device, dtype=torch.float))

		# initialize game
		board = Backgammon.init_board()
		# state = feature_encoding(np.copy(board))
		player = np.random.randint(2)*2-1 # which player begins?

		search(np.copy(board),player)

		# find action for the first player
		dice = Backgammon.roll_dice() # first roll
		a = action(np.copy(board), dice, player, 0)

		first = True
		while not Backgammon.game_over(board) and not Backgammon.check_for_error(board):
			for i in range(1+int(dice[0] == dice[1])):
				count += 1 if i != 1 else count + 0
				if first and dice[0] == dice[1]: i + 1 # if the first roll was double, take into account
				first = False

				#Execute action
				new_board = np.copy(board)
				for m in a:
					new_board = Backgammon.update_board(new_board, m, player)

				reward = 1 if Backgammon.game_over(new_board) else 0

				switch = 1 if i == 0 and dice[0] == dice[1] else -1

				search(np.copy(new_board),player*switch)
				
				#action for next player or same if double roll
				doubleD = 1 if switch == 1 else 0
				
				new_action = action(np.copy(new_board), dice, player*switch,i)
				
				if count > 3:
					print('up')
					if player == 1:
						x = Variable(torch.tensor(feature_encoding(board, doubleD), dtype=torch.float, device = device)).view(197,1)
						y = forward(x)
						yold = forward(xold1)
						delta = reward + y - yold # TD-error
						
						# update theta
						thetaUpdate(z1_w1,z1_b1,z1_w2,z1_b2,z1_w3,z1_b3, delta)
						#elligibillyboy
						z1_w1,z1_b1,z1_w2,z1_b2,z1_w3,z1_b3 = eligibillyUpdate(z1_w1,z1_b1,z1_w2,z1_b2,z1_w3,z1_b3, yold)

					else:
						# Flip?
						flipped = flipped_agent.flip_board(np.copy(board))
						x = Variable(torch.tensor(feature_encoding(flipped, doubleD), dtype=torch.float, device = device)).view(197,1)
						y = forward(x)
						yold = forward(xold2) # xold already flipped
						delta = reward + y - yold # TD-error
						
						# update theta
						thetaUpdate(z2_w1,z2_b1,z2_w2,z2_b2,z2_w3,z2_b3, delta)
						#elligibillyboy
						z2_w1,z2_b1,z2_w2,z2_b2,z2_w3,z2_b3 = eligibillyUpdate(z2_w1,z2_b1,z2_w2,z2_b2,z2_w3,z2_b3, yold)

				if player == 1:
					xold1 = Variable(torch.tensor(feature_encoding(board, doubleD), dtype=torch.float, device = device)).view(197,1)
				else:
					#Flip?
					flipped = flipped_agent.flip_board(np.copy(board))
					xold2 = Variable(torch.tensor(feature_encoding(flipped, doubleD), dtype=torch.float, device = device)).view(197,1)

				board = new_board
				a = new_action

				player = player * switch

def thetaUpdate(zw1, zb1, zw2, zb2, zw3, zb3, delta):
	global w1,b1,w2,b2,w3,b3
	w1 = w1 + 0.01* delta * zw1
	b1 = b1 + 0.01* delta * zb1
	w2 = w2 + 0.01* delta * zw2
	b2 = b2 + 0.01* delta * zb2
	w3 = w3 + 0.01* delta * zw3
	b3 = b3 + 0.01* delta * zb3

def thetaUpdate_trans(zw1, zb1, zw2, zb2, zw3, zb3, delta):
	global w1_trans,b1_trans,w2_trans,b2_trans,w3_trans,b3_trans
	w1_trans = w1_trans + 0.01* delta * zw1
	b1_trans = b1_trans + 0.01* delta * zb1
	w2_trans = w2_trans + 0.01* delta * zw2
	b2_trans = b2_trans + 0.01* delta * zb2
	w3_trans = w3_trans + 0.01* delta * zw3
	b3_trans = b3_trans + 0.01* delta * zb3

def eligibillyUpdate(zw1, zb1, zw2, zb2, zw3, zb3, phi):
	zw1 = 0.7 * zw1 + phi
	zb1 = 0.7 * zb1 + phi
	zw2 = 0.7 * zw2 + phi
	zb2 = 0.7 * zb2 + phi
	zw3 = 0.7 * zw3 + phi
	zb3 = 0.7 * zb3 + phi

	return zw1, zb1, zw2, zb2, zw3, zb3

def load(name):
	try:
		w1 = torch.load(name+'w1')
		w2 = torch.load(name+'w2')
		w3 = torch.load(name+'w3')
		b1 = torch.load(name+'b1')
		b2 = torch.load(name+'b2')
		b3 = torch.load(name+'b3')
	except FileNotFoundError:
		print('starting fresh')

def save(name):
	torch.save(w1, name+'w1')
	torch.save(b1, name+'b1')
	torch.save(w2, name+'w2')
	torch.save(b2, name+'b2')
	torch.save(w3, name+'w3')
	torch.save(b3, name+'b3')
	print('saved to ', name)


def main():
	# load
	load('./dynaW/')
	learn(1)
	save('./dynaW/')
	# learn
	learn(50000)
	# til öryggis
	save('./dynaW/')


if __name__ == '__main__':
	main()
