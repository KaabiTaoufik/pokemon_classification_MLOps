import matplotlib.pyplot as plt

def plot_history(history):
    fig = plt.figure()
    plt.plot(history.history['acc'], color='teal', label='accuracy')
    plt.plot(history.history['loss'], color='orange', label='loss')
    fig.suptitle('Performance', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()