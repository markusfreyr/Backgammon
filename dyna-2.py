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
theta1 = np.zeros(196)
theta_trans1 = np.zeros(196)
theta2 = np.zeros(196)
theta_trans2 = np.zeros(196)

def feature_encoding(board):
	features = np.zeros(196)

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

	return features

def Q_trans(features, player):
	global theta1, theta_trans1, theta2, theta_trans2
	if player == 1:
		return np.dot(np.array(features), theta1) + np.dot(np.array(features), theta_trans1)
	else:
		return np.dot(np.array(features), theta2) + np.dot(np.array(features), theta_trans2)

def action(board, dice, player, doubleD):
	possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player)
	na = np.zeros(len(possible_boards))
	for i in range(possible_boards):
		features = feature_encoding(possible_boards[i])
		na[i] = Q_trans(features, player)
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
		# initialize game
		board = Backgammon.init_board()
		state = feature_encoding(np.copy(board))
		player = np.random.randint(2)*2-1 # which player begins?

		dice = Backgammon.roll_dice() # first roll

		theta_trans1 = np.zeros(196)
		theta_trans2 = np.zeros(196)
		z = 0
		search(board)
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
						heta2 = theta2 + 0.1 * delta * z2
						z2 = 0.1 * z2 + 0.1

				if player == 1:
					xold1 = state
				else:
					xold2 = state

				state = new_state
				board = new_board
				action = new_action

			player = player * -1


def main():
	board = Backgammon.init_board()
	features = feature_encoding(board)
	#Backgammon.pretty_print(board)
	#print(board)
	#print(features)
	''' for i in range(0, 196):
		print(features[i],features[i+1],features[i+2],features[i+3])
		i = i + 4 '''

	a = np.array([2,2,2])
	b = np.array([2,2,2]).transpose()
	print(np.dot(a,b))


if __name__ == '__main__':
	main()
