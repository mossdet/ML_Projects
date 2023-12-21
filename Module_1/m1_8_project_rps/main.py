import random as rd
from rps_ascii_art import *


def get_rps_play(selection):
    rps_play = {'symbol': None, 'name': None, 'idx': None}
    if selection == 1:
        rps_play['symbol'] = rock
        rps_play['name'] = "rock"
        rps_play['idx'] = 1
    elif selection == 2:
        rps_play['symbol'] = paper
        rps_play['name'] = "paper"
        rps_play['idx'] = 2
    elif selection == 3:
        rps_play['symbol'] = scissors
        rps_play['name'] = "scissors"
        rps_play['idx'] = 3

    return rps_play


def determine_rps_winner(user_play, cpu_play):
    # winner variable is a flag:
    # 0 = tie
    # 1 = user
    # 2 = cpu
    user_symbol = user_play['name']
    cpu_symbol = cpu_play['name']

    winner = 0
    if user_symbol != cpu_symbol:
        winner = 2
        if user_symbol == "rock" and cpu_symbol == "scissors":
            winner = 1
        elif user_symbol == "paper" and cpu_symbol == "rock":
            winner = 1
        elif user_symbol == "scissors" and cpu_symbol == "paper":
            winner = 1

    return winner


selection = 1
while selection != 4:
    selection = int(
        input("Select Rock(1), Paper(2), Scissors(3) or Quit(4): "))
    if selection != 4:

        # User Hand
        user_play = get_rps_play(selection)
        if user_play['idx'] == None:
            print(f"Invalid Selection: {selection}")
            continue

        # CPU Hand
        selection = rd.randint(1, 3)
        cpu_play = get_rps_play(selection)
        if cpu_play['idx'] == None:
            print("Invalid CPU Selection {selection}")
            continue

        print(f"User plays {user_play['name']}: \n{user_play['symbol']}")
        print(f"Computer plays {cpu_play['name']}: \n{cpu_play['symbol']}")

        winner = determine_rps_winner(user_play, cpu_play)

        # Show winner and ASCII Art
        if winner == 0:
            print('It\'s a tie!')
            print(tie_emoji)
        elif winner == 1:
            print('You win!')
            print(win_emoji)
        elif winner == 2:
            print('You lose!')
            print(loose_emoji)

    else:
        print("Bye bye, thanks for playing!")
