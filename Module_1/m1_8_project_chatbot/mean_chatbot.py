import random as rd
import chatbot_art
from datetime import datetime


class ChatBot():

    run_chatbot_flag = True
    greet_str = chatbot_art.greet_str
    bot_face = chatbot_art.chatbot_face
    user_win_face = chatbot_art.user_win_face

    def __init__(self, total_tries, total_hints):

        rd.seed(int(datetime.now().timestamp()))

        self.tries_left = total_tries
        self.hints_lift = total_hints
        self.__rand_int = rd.randint(0, 99)

    def get_rand_int(self):
        return self.__rand_int

    def greet_user(self):
        print(self.bot_face)
        print(self.greet_str)
        print(
            f"\t\t You have {self.tries_left} attempts, choose carefully... or you WILL perish!!!\n\n")

    def check_guess(self, guess_val):
        guess_ok = guess_val == self.__rand_int
        return guess_ok

    def decrement_tries_left(self):
        self.tries_left -= 1

    def decrement_hints_left(self):
        self.hints_lift -= 1

    def get_tries_left(self):
        return (self.tries_left)

    def get_hints_left(self):
        return (self.hints_lift)

    def select_hint_category(hint_val):
        pass

    def provide_hint(self, hint_category):
        if hint_category == "a":
            print(self.get_hint_fact_mult())

        elif hint_category == "b":
            print(self.get_hint_bigger_smaller())

        elif hint_category == "c":
            print(self.get_hint_parity())

    def get_hint_parity(self):
        hint_str = "The number is odd"

        if self.__rand_int % 2 == 0:
            hint_str = "The number is even"

        return hint_str

    def get_hint_bigger_smaller(self):
        hint_str = ""
        hint_sel = rd.randint(0, 1)

        larger_num = None
        smaller_num = None
        if self.__rand_int < 99:
            larger_num = rd.randint(self.__rand_int+1, 99)
        if self.__rand_int > 0:
            smaller_num = rd.randint(0, self.__rand_int-1)

        if hint_sel == 0:
            if larger_num is not None:
                hint_str = f"The number is smaller than {larger_num}"
            else:
                hint_str = f"The number is bigger than {smaller_num}"
        else:
            if smaller_num is not None:
                hint_str = f"The number is bigger than {smaller_num}"
            else:
                hint_str = f"The number is smaller than {larger_num}"

        return hint_str

    def get_hint_fact_mult(self):

        hint_str = ""
        hint_type_str = ""
        hint_val = 0
        hint_sel = rd.randint(0, 1)

        if hint_sel == 0:

            factor = self.get_factors()
            hint_type_str = "factor"
            hint_val = factor

            if factor is None:

                multiple = self.get_multiples()
                hint_type_str = "multiple"
                hint_val = multiple

                if multiple is None:
                    hint_type_str = "none"
        else:
            multiple = self.get_multiples()
            hint_type_str = "multiple"
            hint_val = multiple

            if multiple is None:
                factor = self.get_factors()
                hint_type_str = "factor"
                hint_val = factor

                if factor is None:
                    hint_type_str = "none"

        if hint_val is None:
            hint_str = f"The number has no factors or multiples"
        else:
            hint_str = f"A {hint_type_str} of the number is {hint_val}"

        return hint_str

    def get_factors(self):
        rand_factor = None
        factors_ls = []
        for n in range(1, self.__rand_int+1):
            remainder = self.__rand_int % n
            quotient = self.__rand_int / n

            # stop looking for factors when the factors start getting reversed
            if quotient < n:
                break

            if remainder == 0:
                factors_ls.append(n)
                factors_ls.append(quotient)

        factors_ls.sort()
        if len(factors_ls) > 0:
            rand_idx = rd.randint(0, len(factors_ls)-1)
            rand_factor = factors_ls[rand_idx]

        return rand_factor

    def get_multiples(self):
        multiples_ls = []
        for n in range(1, 11):
            mult_val = n*self.__rand_int
            multiples_ls.append(mult_val)

        if len(multiples_ls) > 0:
            rand_idx = rd.randint(0, len(multiples_ls)-1)
            rand_multpl = multiples_ls[rand_idx]
            return rand_multpl
        else:
            return None
