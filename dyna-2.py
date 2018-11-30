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

def Q(features, player):
	global theta1, theta_trans1, theta2, theta_trans2
	if player == 1:
		return np.dot(np.array(features), theta1)
	else:
		return np.dot(np.array(features), theta2)


def action(board, dice, player, doubleD):
	print("board")
	print(board)
	print(dice)
	print(Backgammon.check_for_error(board))
	#if Backgammon.check_for_error(board):
	#	k = k+1
	possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player)
	print("test")
	if possible_moves == []:
		return []
	na = np.zeros(len(possible_boards))
	for i in range(len(na)):
		features = feature_encoding(possible_boards[i])
		na[i] = Q_trans(features, player)
	return possible_moves[np.argmax(na)]

def search(board,player):
	global theta1, theta_trans1, theta2, theta_trans2
	start = time.time()
	time_available = 1
	original_board = np.copy(board)
	if Backgammon.game_over(board):
		return
	while time_available - (time.time() - start) > 0:
		board = np.copy(original_board)
		z_trans1 = np.zeros(196)
		z_trans2 = np.zeros(196)

		dice = Backgammon.roll_dice()
		a1 = action(np.copy(board), dice, player, 0)
		if player == 1:
			board1_old = np.copy(board)
		else:
			board2_old = np.copy(board)
		for i in range(1+int(dice[0] == dice[1])):
			# update the board
			if len(a1) != 0:
				for m in a1:
					board = Backgammon.update_board(board, m, player)
		reward = 0
		if Backgammon.game_over(board):
			return

		player = player*-1
		dice = Backgammon.roll_dice()
		a2 = action(np.copy(board), dice, player, 0)
		if player == 1:
			board1_old = np.copy(board)
		else:
			board2_old = np.copy(board)
		for i in range(1+int(dice[0] == dice[1])):
			# update the board
			if len(a2) != 0:
				for m in a2:
					board = Backgammon.update_board(board, m, player)
		
		player = player*-1
		dice = Backgammon.roll_dice()
		a = action(np.copy(board), dice, player, 0)
		if Backgammon.game_over(board):
			return
		counting_now = 0
		while not Backgammon.game_over(board):
			print("count")
			print(counting_now)
			counting_now += 1

			print()
			print(board)
			print(a)


			for i in range(1+int(dice[0] == dice[1])):
				board_new = np.copy(board) 
			
				# update the board
				if len(a) != 0:
					for m in a:
						print(m)
						board_new = Backgammon.update_board(np.copy(board_new), m, player)
						print(Backgammon.check_for_error(board_new))
			
			if Backgammon.check_for_error(board_new):
				exit()


			#board_new = A(board,a)
			#reward = B(board,a)

			if Backgammon.game_over(board_new):
				reward = 1
			else:
				reward = 0

			player = player*-1
			
			dice = Backgammon.roll_dice()
			a_new = action(np.copy(board_new), dice, player, 0)
			feature_new = feature_encoding(np.copy(board_new))

			if player == 1:
				feature_old = feature_encoding(np.copy(board1_old))
			else:
				feature_old = feature_encoding(np.copy(board2_old))
			delta_trans = reward + Q_trans(feature_new,player) - Q_trans(feature_old,player)

			if player == 1:
				theta_trans1 = theta_trans1 + 0.01*delta_trans*z_trans1
				z_trans1 = 0.5*z_trans1 + feature_old
			else:
				theta_trans2 = theta_trans2 + 0.01*delta_trans*z_trans2
				z_trans2 = 0.5*z_trans2 + feature_old
			
			if player == 1:
				board1_old = np.copy(board)
			else:
				board2_old = np.copy(board)

			board = board_new
			a = a_new

def learn(n):
	global theta1, theta_trans1, theta2, theta_trans2

	A = defaultdict(list)
	B = defaultdict(list)
	#theta = 0 gert annarsstaðar
	#while True:
	for i in range(n):
		print(i)
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
		search(board,player) #senda inn A,B og player inn í search ?? ?? ?? ??
		a = action(np.copy(board), dice, player, 0)

		while not Backgammon.game_over(board):
			for i in range(1+int(dice[0] == dice[1])):
				count += 1 if i != 1 else count + 0
				#Execute action
				new_board = []
				for m in a:
					new_board = Backgammon.update_board(board, m, player)
				#Observe state for nextplayer
				new_state = feature_encoding(np.copy(new_board))
				#Observe reward
				reward = 1 if Backgammon.game_over(new_board) else 0

				# update model
				#A[(state,a)] = new_board
				#B[(state,a)] = reward
				# search with new state
				search(new_board,player)
				
				#action for the next player?
				dice = Backgammon.roll_dice()
				new_action = action(new_board, dice, player*-1,0)
				
				if count > 3:
					if player == 1:
						delta = reward + Q(state,player) - Q(xold1,player)
						#theta1 = theta1 + alpha(board,a)*delta*z
						#z = lamb*z + phi
						theta1 = theta1 + 0.1 * delta * z1
						z1 = 0.1 * z1 + 0.1
					else:
						delta = reward + Q(state,player) - Q(xold2,player)
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
				a = new_action

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

	learn(10)


if __name__ == '__main__':
	main()
