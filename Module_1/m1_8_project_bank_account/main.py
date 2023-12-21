'''
'''
# Import date class from datetime module
from datetime import datetime
import random as rd
from datapaths import *


class BankAccount():

    def __init__(self, account_type, account_nr, account_holder, history_path):
        self.account_type = account_type
        self.account_nr = account_nr
        self.account_holder = account_holder
        self._balance = 0
        self.filename = history_path + str(self.account_nr)+"_" + \
            self.account_type+"_"+self.account_holder+".txt"

        # Save account creation event to file
        history_entree = self.get_timenow_str()+"\t"+"Account Creation"+"\t" +\
            account_type+"\t" + str(account_nr)+"\t"+account_holder+"\t" + \
            "Balance="+"\t" + str(self.get_balance())+"\n"
        history_file = open(self.filename, 'w')
        history_file.write(history_entree)
        history_file.close()

    def get_timenow_str(self):
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d, %H:%M:%S")
        return date_time

    def refresh_history(self, entree):
        history_file = open(self.filename, 'a')
        history_file.write(entree)
        history_file.close()

    def read_history(self):
        history_file = open(self.filename, 'r')
        history_ls = history_file.readlines()
        history_file.close()
        return history_ls

    def deposit(self, amount):
        self._balance += amount
        history_entree = self.get_timenow_str()+"\t"+"Deposit="+"\t" +\
            str(amount)+"\t"+"Balance="+"\t" + str(self.get_balance())+"\n"
        self.refresh_history(history_entree)

    def withdraw(self, amount):
        if self.get_balance() >= amount:
            self._balance -= amount
            history_entree = self.get_timenow_str()+"\t"+"Withdraw="+"\t" +\
                str(amount)+"\t"+"Balance="+"\t" + str(self.get_balance())+"\n"
            self.refresh_history(history_entree)
        else:
            history_entree = self.get_timenow_str()+"\t"+"Withdrawal Limit Exceeded="+"\t" +\
                str(-1*amount)+"\t"+"Balance=" + \
                "\t" + str(self.get_balance())+"\n"
            self.refresh_history(history_entree)

    def get_balance(self):
        return self._balance

    def get_account_type(self):
        return self.account_type

    def get_account_nr(self):
        return self.account_nr

    def get_account_nr(self):
        return self.account_nr

    def get_account_holder(self):
        return self.account_holder


def generate_account_nr():
    accont_nr = rd.randint(0, 9)
    for digit in range(1, 10):
        val = (10**digit) * rd.randint(0, 9)
        accont_nr += val
        pass
    return accont_nr


# Test 1
accont_nr = generate_account_nr()
account_1 = BankAccount("Checking Account", accont_nr,
                        'Ronaldo da Silva Souza', history_path)
account_1.deposit(1000000)
account_1.withdraw(800000)
assert (account_1.get_balance() == 200000), "Accounting Error"
account_1.withdraw(800000)
account_1.deposit(523000.15)
assert (account_1.get_balance() == 723000.15), "Accounting Error"
account_1.withdraw(700000)
account_1.deposit(250000)

# Print account info
print("Account Type = ", account_1.get_account_type())
print("Account Number = ", account_1.get_account_nr())
print("Account Holder Name = ", account_1.get_account_holder())
print("Balance = ", account_1.get_balance())
print("Transaction History = ")
history_ls = account_1.read_history()
for entree in history_ls:
    print(entree)

pass

# Test 2
accont_nr = generate_account_nr()
account_2 = BankAccount("Savings Account", accont_nr,
                        'Cristiano Ronaldo', history_path)
account_2.deposit(1200)
account_2.withdraw(1100)
assert (account_2.get_balance() == 100), "Accounting Error"
account_2.withdraw(800)
account_2.deposit(125.37)
assert (account_2.get_balance() == 225.37), "Accounting Error"
account_2.withdraw(200.37)
account_2.deposit(25)
assert (account_2.get_balance() == 50), "Accounting Error"

# Print account info
print("Account Type = ", account_2.get_account_type())
print("Account Number = ", account_2.get_account_nr())
print("Account Holder Name = ", account_2.get_account_holder())
print("Balance = ", account_2.get_balance())
print("Transaction History = ")
history_ls = account_2.read_history()
for entree in history_ls:
    print(entree)

pass
