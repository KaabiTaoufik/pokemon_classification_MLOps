import os
os.chdir('data')
os.system('kaggle datasets download -d bhawks/pokemon-generation-one-22k && unzip \*.zip && rm \*.zip')


