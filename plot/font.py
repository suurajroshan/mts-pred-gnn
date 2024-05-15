import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.font_manager import FontProperties


def get_font():
    fontManager.addfont('./times+simsun.ttf')
    font = FontProperties(fname='./times+simsun.ttf')
    plt.rcParams['font.sans-serif'] = font.get_name()
    plt.rcParams['font.family'] = 'sans-serif'
    return font

