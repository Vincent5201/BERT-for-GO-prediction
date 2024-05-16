import pygame
import sys
import time
import copy
import random
from use import vote_next_move
import numpy as np

from tools import Lchannel_01, transfer_back

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BROWN = (165, 42, 42)
TRANSPARENT_BLACK = (0, 0, 0, 20)  
TRANSPARENT_WHITE = (255, 255, 255, 20) 
BACKGROUND_COLOR = (173, 216, 230)
BUTTON_COLOR = (100, 100, 100)

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 660
GRID_SIZE = 18
GRID_WIDTH = 650 // GRID_SIZE
GRID_HEIGHT = 650 // GRID_SIZE
LEFT_TOP = 10

INFO_BOX_WIDTH = 150
INFO_BOX_HEIGHT = 50
INFO_BOX_X = 700
INFO_BOX_Y = 130

BUTTON_WIDTH = 80
BUTTON_HEIGHT = 30
BUTTON_X = INFO_BOX_X
BUTTON_Y = INFO_BOX_Y + INFO_BOX_HEIGHT + 20
BUTTON_COOLDOWN = 0.5
RADIUS = 10

running = True
turn = 1
board = [[[[0]*(GRID_SIZE + 1) for _ in range(GRID_SIZE + 1)] for _ in range(4)]]
board = np.array(board)
board_history = []
text = ""
games = []
last_move = [-1, -1]
mode = "standby"
button_cool = True
cool_time = 0
turn = 1
playing = False
computer_turn = 0

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("圍棋")

def draw_board(board):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(LEFT_TOP + x * GRID_WIDTH, LEFT_TOP + y * GRID_HEIGHT,
                                GRID_WIDTH, GRID_HEIGHT)
            pygame.draw.rect(screen, BROWN, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)  
    for i in range(GRID_SIZE + 1):
        for j in range(GRID_SIZE + 1):
            if board[0][0][i][j]:
                pygame.draw.circle(screen, TRANSPARENT_WHITE,
                                   (LEFT_TOP + j * GRID_WIDTH, LEFT_TOP + i * GRID_HEIGHT), RADIUS)
            if board[0][1][i][j]:
                pygame.draw.circle(screen, TRANSPARENT_BLACK,
                                   (LEFT_TOP + j * GRID_WIDTH, LEFT_TOP + i * GRID_HEIGHT), RADIUS)

def get_board_position(mouse_pos):
    grid_y = (mouse_pos[0] - LEFT_TOP + GRID_WIDTH // 2) // GRID_WIDTH
    grid_x = (mouse_pos[1] - LEFT_TOP + GRID_WIDTH // 2) // GRID_HEIGHT 
    return grid_x, grid_y

def draw_info_box(coordinates, x, y, width, height):
    info_box_rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(screen, WHITE, info_box_rect)
    pygame.draw.rect(screen, BLACK, info_box_rect, 2)

    font = pygame.font.SysFont(None, 24)
    text = font.render(coordinates, True, BLACK)
    text_rect = text.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text, text_rect)

def draw_button(text, x, y, width, height, action):
    button_rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(screen, BUTTON_COLOR, button_rect)

    font = pygame.font.SysFont(None, 24)
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text_surface, text_rect)

    mouse_pos = pygame.mouse.get_pos()
    if button_rect.collidepoint(mouse_pos):
        if pygame.mouse.get_pressed()[0]:
            action()

def reset_game():
    global button_cool
    if button_cool:
        global mode, turn, board, text, games, board_history, cool_time, playing
        button_cool = False
        cool_time = time.time()
        mode = "standby"
        turn = 1
        board = [[[[0]*(GRID_SIZE + 1) for _ in range(GRID_SIZE + 1)] for _ in range(4)]]
        board = np.array(board)
        text = ""
        games = []
        board_history = []
        playing = False

def back():
    global button_cool
    if button_cool:
        global games, turn, board, board_history, cool_time
        button_cool = False
        cool_time = time.time()
        board = board_history[-2]
        board_history = board_history[:-1]
        turn = 0 if turn else 1
        games = games[:-1]
        
def start():
    global button_cool
    if button_cool:
        global cool_time
        button_cool = False
        cool_time = time.time()
        global text, computer_turn, turn, board, text, games, board_history, playing, mode
        if len(games):
            text = "reset first"
            return
        mode = "playing"
        computer_turn = random.randint(0,1)
        playing = True

    
def quit_game():
    pygame.quit()
    sys.exit()

while running:
    screen.fill(BACKGROUND_COLOR)
    if time.time()-cool_time > BUTTON_COOLDOWN:
        button_cool = True
    if playing and computer_turn == turn:
        results, moves = vote_next_move([games], "cpu")
        tgt = 0
        while moves[tgt] == last_move[turn] or board[0][0][results[tgt][0]][results[tgt][1]] \
                            or board[0][1][results[tgt][0]][results[tgt][1]]:
            tgt += 1
        result = results[tgt]
        move = moves[tgt]
        last_move[turn] = move
        Lchannel_01(board, 0, result[0], result[1], turn)
        text = f"{result[0]}, {result[1]}"
        games.append(move)
        board_history.append(copy.deepcopy(board))
        turn = 0 if turn else 1
    for event in pygame.event.get():
        mouse_pos = pygame.mouse.get_pos()
        grid_x, grid_y = get_board_position(mouse_pos)
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if grid_x >= 0 and grid_y >= 0 and grid_x <= GRID_SIZE and grid_y <= GRID_SIZE:
                if board[0][0][grid_x][grid_y] or board[0][1][grid_x][grid_y]:
                    text = "invalid position"
                else:
                    Lchannel_01(board, 0, grid_x, grid_y, turn)
                    text = f"{grid_x}, {grid_y}"
                    games.append(transfer_back(grid_x*19+grid_y))
                    board_history.append(copy.deepcopy(board))
                    turn = 0 if turn else 1
                
    draw_board(board)
    
    draw_info_box(text, INFO_BOX_X, INFO_BOX_Y, INFO_BOX_WIDTH, INFO_BOX_HEIGHT)
    draw_info_box(mode, INFO_BOX_X, INFO_BOX_Y - INFO_BOX_HEIGHT - 10, INFO_BOX_WIDTH, INFO_BOX_HEIGHT)
    
    draw_button("Reset", BUTTON_X, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT, reset_game)
    draw_button("Back", BUTTON_X, BUTTON_Y + BUTTON_HEIGHT + 10, BUTTON_WIDTH, BUTTON_HEIGHT, back)
    draw_button("Quit", BUTTON_X, BUTTON_Y + 2 * (BUTTON_HEIGHT + 10), BUTTON_WIDTH, BUTTON_HEIGHT, quit_game)
    draw_button("Start", BUTTON_X, BUTTON_Y + 3 * (BUTTON_HEIGHT + 10), BUTTON_WIDTH, BUTTON_HEIGHT, start)
    

    if grid_x >= 0 and grid_y >= 0 and grid_x <= GRID_SIZE and grid_y <= GRID_SIZE:
        
        if turn:
            pygame.draw.circle(screen, TRANSPARENT_BLACK, mouse_pos, RADIUS)
        else:
            pygame.draw.circle(screen, TRANSPARENT_WHITE, mouse_pos, RADIUS)


    pygame.display.flip()

quit_game()
