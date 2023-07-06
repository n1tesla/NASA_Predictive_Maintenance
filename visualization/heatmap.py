import seaborn as sns

def heatmap(df):
    sns.heatmap(df.corr(),annot=True,cmap="RdY1Gn",linewidths=0.2)
    fig=plt.gcf()
    fig.set_size_inches(20,20)
    return plt.show()
