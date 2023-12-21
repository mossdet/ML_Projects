import mean_chatbot
import chatbot_art


bot = mean_chatbot.ChatBot(4, 3)

# Greet the human
bot.greet_user()

# print(bot.get_rand_int())

game_on = True
while game_on:
    # Ask for a guess
    guess_val = input("What is your guess human?: ")

    if guess_val == "hint":

        if bot.get_hints_left() > 0:
            # Handle asking hint
            chatbot_art.make_space()
            hint_category = input(
                "Choose a hint category:\n(a)Factors, multiples\n(b)Larger,Smaller\n(c)Parity:\n")
            chatbot_art.make_space()
            while hint_category not in ["a", "b", "c"]:
                hint_category = input(
                    "Invalid choice.\n Choose a hint category:\n(a)Factors, multiples\n(b)Larger,Smaller\n(c)Parity:\n")
                chatbot_art.make_space()

            bot.decrement_hints_left()
            bot.provide_hint(hint_category)
            print(f"You have {bot.get_hints_left()} hints left")
        else:
            print("You have no hints left")

        chatbot_art.make_space()

    elif not guess_val.isnumeric():
        # Handle invalid input
        print("Invalid choice, guess again.\n")

    else:
        # Handle valid guess

        guess_val = int(guess_val)

        # Check if guess is correct
        guess_ok = bot.check_guess(guess_val)
        chatbot_art.make_space()

        if guess_ok:
            print(chatbot_art.user_win_face)
            print("You won!!!\nYou were very lucky, I'll get you next time though!")
            game_on = False
        else:
            bot.decrement_tries_left()
            if bot.get_tries_left() <= 0:
                print(chatbot_art.chatbot_face)
                print(
                    "You loose! You ran out of tries and the world will end soon because of you")
                print(
                    f"The number was {bot.get_rand_int()}, you can go home and cry now.")
                game_on = False

            else:
                print(
                    f"You guessed wrong. You have {bot.get_tries_left()} tries left")
                print(
                    f"Type \"hint\" and I will provide you with a hint, you have {bot.get_hints_left()} hints left")
                chatbot_art.make_space()
