import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(board, side, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(board, side, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (moveList, moveTree, value)
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
      value (int or float): value of the board after making the chosen move
    Input:
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      side (boolean): True if player1 (Min) plays next, otherwise False
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(board, side, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return ([ move ], { encode(*move): {} }, value)
    else:
        return ([], {}, evaluate(board))

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.

def minimax(board, side, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (moveList, moveTree, value)
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
      value (float): value of the final board in the minimax-optimal move sequence
    Input:
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      side (boolean): True if player1 (Min) plays next, otherwise False
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''

    # making moveList global doesn't work
    # appending submoveList causes program to crash
    moveList = []
    moveTree = {}
    value = 0
    
    boardValues = {}
    moves = [ move for move in generateMoves(board, side, flags) ]
    if depth == 0 or len(moves) == 0:
        return ([], {}, evaluate(board))
    if not side: # maximizingPlayer
        value = float('-inf')
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])

            submoveList, submoveTree, subvalue = minimax(newboard, newside, newflags, depth - 1)

            value = max(value, subvalue)

            move_encoded = encode(move[0], move[1], move[2])
            boardValues[move_encoded] = value
            moveTree[move_encoded] = submoveTree


        best_move = max(boardValues, key=boardValues.get)
        decoded_best_move = decode(best_move)
        newside, newboard, newflags = makeMove(side, board, decoded_best_move[0], decoded_best_move[1], flags, decoded_best_move[2])
        bestmoveList = minimax(newboard, newside, newflags, depth - 1)[0]

        return ([decoded_best_move] + bestmoveList, moveTree, value)
    else: # minimizingPlayer
        value = float('inf')
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])

            submoveList, submoveTree, subvalue = minimax(newboard, newside, newflags, depth - 1)

            value = min(value, subvalue)


            move_encoded = encode(move[0], move[1], move[2])
            boardValues[move_encoded] = value
            moveTree[move_encoded] = submoveTree


        best_move = min(boardValues, key=boardValues.get)
        decoded_best_move = decode(best_move)
        newside, newboard, newflags = makeMove(side, board, decoded_best_move[0], decoded_best_move[1], flags, decoded_best_move[2])
        bestmoveList = minimax(newboard, newside, newflags, depth - 1)[0]

        return ([decoded_best_move] + bestmoveList, moveTree, value)

    #raise NotImplementedError("you need to write this!")

def alphabeta(board, side, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (moveList, moveTree, value)
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
      value (float): value of the final board in the minimax-optimal move sequence
    Input:
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      side (boolean): True if player1 (Min) plays next, otherwise False
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    moveList = []
    moveTree = {}
    value = 0
    
    boardValues = {}
    moves = [ move for move in generateMoves(board, side, flags) ]
    if depth == 0 or len(moves) == 0:
        return ([], {}, evaluate(board))
    if not side: # maximizingPlayer
        value = float('-inf')
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])

            submoveList, submoveTree, subvalue = alphabeta(newboard, newside, newflags, depth - 1, alpha, beta)

            value = max(value, subvalue)

            move_encoded = encode(move[0], move[1], move[2])
            boardValues[move_encoded] = value
            moveTree[move_encoded] = submoveTree

            alpha = max(value, alpha)

            if alpha >= beta:
                break

        best_move = min(boardValues, key=boardValues.get)
        decoded_best_move = decode(best_move)
        newside, newboard, newflags = makeMove(side, board, decoded_best_move[0], decoded_best_move[1], flags, decoded_best_move[2])
        bestmoveList = minimax(newboard, newside, newflags, depth - 1)[0]

        return ([decoded_best_move] + bestmoveList, moveTree, value)
    else: # minimizingPlayer
        value = float('inf')
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])

            submoveList, submoveTree, subvalue = alphabeta(newboard, newside, newflags, depth - 1, alpha, beta)

            value = min(value, subvalue)

            move_encoded = encode(move[0], move[1], move[2])
            boardValues[move_encoded] = value
            moveTree[move_encoded] = submoveTree

            beta = min(value, beta)

            if alpha >= beta:
                break

        best_move = min(boardValues, key=boardValues.get)
        decoded_best_move = decode(best_move)
        newside, newboard, newflags = makeMove(side, board, decoded_best_move[0], decoded_best_move[1], flags, decoded_best_move[2])
        bestmoveList = minimax(newboard, newside, newflags, depth - 1)[0]

        return ([decoded_best_move] + bestmoveList, moveTree, value)

    #raise NotImplementedError("you need to write this!")
    

def stochastic(board, side, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (moveList, moveTree, value)
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
      value (float): average board value of the paths for the best-scoring move
    Input:
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      side (boolean): True if player1 (Min) plays next, otherwise False
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    raise NotImplementedError("you need to write this!")
