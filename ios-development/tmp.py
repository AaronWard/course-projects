import os


for i in range(1, 36):
    os.mkdir(os.path.join(os.getcwd(), 'section_{}'.format(i)))