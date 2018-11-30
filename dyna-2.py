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
from torch.autograd import Variable
device = torch.device('cpu')

# Define theta and theta_trans, implement load later
w1 = Variable(torch.randn(4*24,197, device = device, dtype=torch.float))
b1 = Variable(torch.zeros((4*24,1), device = device, dtype=torch.float))
w2 = Variable(torch.randn(2*24,4*24, device = device, dtype=torch.float))
b2 = Variable(torch.zeros((2*24,1), device = device, dtype=torch.float))
w3 = Variable(torch.randn(1,2*24, device = device, dtype=torch.float))
b3 = Variable(torch.zeros((1,1), device = device, dtype=torch.float))

def forward(x):
	h = (torch.mm(w1,x) + b1 ).sigmoid()
	#h_sigmoid = h.sigmoid() 
	h2 = (torch.mm(w2,h) + b2).sigmoid()
	#h2_sigmoid = h2.sigmoid()
	y = torch.mm(w3,h2) + b3
	#y_sigmoid = y.sigmoid()
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


def Q_trans(features, player):
	global theta1, theta_trans1, theta2, theta_trans2
	if player == 1:
		return np.dot(np.array(features), theta1) + np.dot(np.array(features), theta_trans1)
	else:
		return np.dot(np.array(features), theta2) + np.dot(np.array(features), theta_trans2)

def action(board, dice, player, doubleD):
	double = 1 if (dice[0]==dice[1] and i == 0) else 0
	if player == -1: board_copy = flipped_agent.flip_board(np.copy(board_copy))
	
	# check out the legal moves available for the throw
	possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player=1)

	# if there are no moves available
	if len(possible_moves) == 0:
			return []

	va = np.zeros(len(possible_boards))

	for i, possibleBoard in enumerate(possible_boards):
		x = Variable(torch.tensor(feature_encoding(possibleBoard), dtype=torch.float, device = device)).view(197,1)
		va[i] = forward(x)

	move = possible_moves[np.argmax(va)]

	if player == -1: move = flipped_agent.flip_move(move)
	
	return move



	
	return possible_moves[np.argmax(na)]

def search(board):
	start = time.time()
	while time_available - (start - time.time()) > 0:
		z_trans = 0
		a = action_trans(board,Q_trans)
		while not Backgammon.game_over(board):
			board_new = A(board,a)
			reward = B(board,a)
			a_new = action(board_new,Q_trans)
			delta_trans = reward + Q_trans(board_new,a_new) - Q_trans(board,a)
			theta_trans = theta_trans + alpha_trans(board,a)*delta_trans*z_trans
			z_trans = lamb_trans*z_trans + phi_trans
			board = new_board
			a = a_new

def learn(n):
	global theta, theta_trans

	A = defaultdict(list)
	B = defaultdict(list)
	#theta = 0 gert annarsstaðar
	#while True:
	for i in range(n):
		count = 0
		xold1 = []
		xold2 = []
		# initialize game
		board = Backgammon.init_board()
		state = feature_encoding(np.copy(board))
		player = np.random.randint(2)*2-1 # which player begins?

		dice = Backgammon.roll_dice() # first roll

		theta_trans1 = np.zeros(196)
		theta_trans2 = np.zeros(196)
		z1 = 0
		z2 = 0
		search(board) #senda inn A,B og player inn í search ?? ?? ?? ??
		action = action(np.copy(board), dice, player, 0)

		while not Backgammon.game_over(board):
			for i in range(1+int(dice[0] == dice[1])):
				count += 1 if i != 1 else count + 0
				#Execute action
				new_board = []
				for m in action:
					new_board = Backgammon.update_board(board, m, player)
				#Observe state for nextplayer
				new_state = feature_encoding(np.copy(new_board))
				#Observe reward
				reward = 1 if Backgammon.game_over(new_board) else 0

				# update model
				A[(state,action)] = new_board
				B[(state,action)] = reward
				# search with new state
				search(new_state)
				
				#action for the next player?
				dice = Backgammon.roll_dice()
				new_action = action(new_board, dice, player*-1)
				
				if count > 2:
					if player == 1:
						delta = reward + Q_trans(state) - Q_trans(xold1)
						#theta1 = theta1 + alpha(board,a)*delta*z
						#z = lamb*z + phi
						theta1 = theta1 + 0.1 * delta * z1
						z1 = 0.1 * z1 + 0.1
					else:
						delta = reward + Q_trans(state) - Q_trans(xold2)
						#theta2 = theta2 + alpha(board,a)*delta*z
						#z = lamb*z + phi
						theta2 = theta2 + 0.1 * delta * z2
						z2 = 0.1 * z2 + 0.1

				if player == 1:
					xold1 = state
				else:
					xold2 = state

				state = new_state
				board = new_board
				action = new_action

			player = player * -1

def load(name):
	w1 = torch.load(name)
	w2 = torch.load(name)
	w3 = torch.load(name)
	b1 = torch.load(name)
	b2 = torch.load(name)
	b3 = torch.load(name)

def save(name):
	torch.save(w1, name)
	torch.save(b1, name)
	torch.save(w2, name)
	torch.save(b2, name)
	torch.save(w3, name)
	torch.save(b3, name)


def main():
	# load
	# Run real exp to initialize model
	# learn
	# save


if __name__ == '__main__':
	main()
