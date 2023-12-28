import os

path = os.path.dirname(os.path.abspath(__file__))
workspacePath = path

history_path = workspacePath + os.path.sep + 'History' + os.path.sep
os.makedirs(history_path, exist_ok=True)
