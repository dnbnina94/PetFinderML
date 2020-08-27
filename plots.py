import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import PercentFormatter
from wordcloud import WordCloud
import math

# print(train["Media"].value_counts())

def bar(dataset,column,xlabel,title,hue=None,rotation=0,showPercentages=True,showCount=False,size="medium",endcodeLabels=True):
    # plt.figure(figsize=(14, 6));

    df = dataset[column].value_counts().sort_index();
    ax = df.plot(kind="bar");

    plt.title(title);

    if endcodeLabels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, horizontalalignment='right');
    else:
        ind = [x for x, _ in enumerate(df)];
        ax.set_xticklabels(ind, rotation=rotation, horizontalalignment='right');

    if showPercentages:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{0:.2%}'.format(height/float(len(dataset))),
                    ha="center",size=size);

    if showCount:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    height,
                    ha="center",size=size);


    plt.ylabel("Count");
    plt.xlabel(xlabel);

def prop_stacked_bar(dataset,featureX,featureY,ylabel,xlabel,title,rotate=False,endcodeLabels=True):
    # plt.figure(figsize=(14, 6));

    valCountsNormalized = [];
    valCounts = [];

    df = dataset.groupby(featureX)[featureY].value_counts().unstack().sort_index();

    ftsX = [];
    for item in df.iterrows():
        ftsX.append(item[0]);
    ind = [x for x, _ in enumerate(ftsX)];

    for item in df.iteritems():
        valCounts.append(np.nan_to_num(np.array(item[1])));

    total = np.sum(valCounts, axis=0);

    for valCount in valCounts:
        valCountsNormalized.append(np.true_divide(valCount, total)*100);

    for i,valCountNormalized in enumerate(valCountsNormalized):
        bottom = np.sum(valCountsNormalized[i+1:],axis=0);
        plt.bar(ind, valCountNormalized, label=i, bottom=bottom);

    if endcodeLabels:
        plt.xticks(ind, ftsX);
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc="upper right")
    plt.title(title);

    if rotate:
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right');

def basic_plot(dataset,column,ylabel,xlabel,title):
    df = dataset[column].value_counts().sort_index();
    ax = df.plot();

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title);

def wordcloud_plot(data,title,freq=False):
    if not freq:
        wordcloud = WordCloud(max_font_size=None, background_color='white').generate(data)
    else:
        wordcloud = WordCloud(max_font_size=None, background_color='white').generate_from_frequencies(data)

    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis("off")

sns.set();

train = pd.read_csv("train_prep.csv");
test = pd.read_csv("test_prep.csv");

fig = plt.figure(figsize=(14, 6));
bar(train,'AdoptionSpeed','Adoption Speed','Adoption speed classes rates');
fig.canvas.set_window_title('Adoption Speed Rate')

fig = plt.figure(figsize=(14, 6));
plt.subplot2grid((1, 3), (0, 0), colspan=1);
bar(train,'Type', 'Pet Type', 'Dogs vs Cats');
plt.subplot2grid((1,3), (0, 1), colspan=1);
prop_stacked_bar(train,'Type', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Type', 'Adoption Speed Rate wrt Pet Type');
plt.subplot2grid((1,3), (0, 2), colspan=1);
bar(test,'Type', 'Pet Type (Test Set)', 'Dogs vs Cats (Test Set)');
fig.canvas.set_window_title('Pet Type')

fig = plt.figure(figsize=(14, 6));
plt.subplot2grid((2,3), (0, 0), colspan=1);
bar(train,'State','State','States',rotation=45,size="xx-small");
plt.subplot2grid((2,3), (0, 1), colspan=1);
prop_stacked_bar(train, 'State', 'AdoptionSpeed', 'Adoption Speed Rate', 'State', 'Adoption Speed Rate wrt State',rotate=True);
plt.subplot2grid((2,3), (0, 2), colspan=1);
bar(test,'State','State','States (Test Set)',rotation=45,size="xx-small");
plt.subplot2grid((2,3), (1, 0), colspan=1);
bar(train,'StateCat','State','State Categories',rotation=45,size="xx-small");
plt.subplot2grid((2,3), (1, 1), colspan=1);
prop_stacked_bar(train, 'StateCat', 'AdoptionSpeed', 'Adoption Speed Rate', 'State', 'Adoption Speed Rate wrt State',rotate=True);
plt.tight_layout()
fig.canvas.set_window_title('State')

fig = plt.figure(figsize=(14, 6));
plt.subplot2grid((2, 4), (0, 0), colspan=1);
bar(train, 'HasName', 'Pet Has Name', 'Pet Has Name');
plt.subplot2grid((2,4), (0, 1), colspan=1);
prop_stacked_bar(train, 'HasName', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Has Name', 'Adoption Speed Rate wrt Pet Has Name');
plt.subplot2grid((2,4), (0, 2), colspan=1);
bar(test, 'HasName', 'Pet Has Name (Test Set)', 'Pet Has Name (Test Set)');
plt.subplot2grid((2,4), (0, 3), colspan=1);
bar(train, 'BadName', 'Pet Has Bad Name', 'Pet Has Bad Name');
plt.subplot2grid((2,4), (1, 0), colspan=1);
prop_stacked_bar(train, 'BadName', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Has Bad Name', 'Adoption Speed Rate wrt Pet Has Bad Name');
plt.subplot2grid((2,4), (1, 1), colspan=1);
wordcloud_plot(' '.join(train[train['Name'].str.len() <= 2]['Name'].fillna('').values),'Bad Names')
plt.subplot2grid((2,4), (1, 2), colspan=1);
wordcloud_plot(' '.join(train.loc[train['Type'] == 'cats', 'Name'].fillna('').values),'Top Cat Names')
plt.subplot2grid((2,4), (1, 3), colspan=1);
wordcloud_plot(' '.join(train.loc[train['Type'] == 'dogs', 'Name'].fillna('').values),'Top Dog Names')
plt.tight_layout()
fig.canvas.set_window_title('Pet Name')

fig = plt.figure(figsize=(14, 6));
plt.subplot2grid((1, 2), (0, 0), colspan=1);
bar(train,'Gender', 'Pet Gender', 'Pet Gneder');
plt.subplot2grid((1,2), (0, 1), colspan=1);
prop_stacked_bar(train, 'Gender', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Gender', 'Adoption Speed Rate wrt Pet Gender');
fig.canvas.set_window_title('Pet Gender')

fig = plt.figure(figsize=(14, 6));
plt.subplot2grid((2,3), (0, 0), colspan=1);
bar(train,'Vaccinated', 'Pet Is Vaccinated', 'Pet Is Vaccinated');
plt.subplot2grid((2,3), (0, 1), colspan=1);
prop_stacked_bar(train, 'Vaccinated', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Is Vaccinated', 'Adoption Speed Rate wrt Pet Is Vaccinated');
plt.subplot2grid((2,3), (0, 2), colspan=1);
bar(train,'Dewormed', 'Pet Is Dewormed', 'Pet Is Dewormed');
plt.subplot2grid((2,3), (1, 0), colspan=1);
prop_stacked_bar(train, 'Dewormed', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Is Dewormed', 'Adoption Speed Rate wrt Pet Is Dewormed');
plt.subplot2grid((2,3), (1, 1), colspan=1);
bar(train,'Sterilized', 'Pet Is Sterilized', 'Pet Is Sterilized');
plt.subplot2grid((2,3), (1, 2), colspan=1);
prop_stacked_bar(train, 'Sterilized', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Is Sterilized', 'Adoption Speed Rate wrt Pet Is Sterilized');
plt.tight_layout()
fig.canvas.set_window_title('Vaccinated, Dewormed, Sterilized')

fig = plt.figure(figsize=(14, 6));
plt.subplot2grid((1,3), (0, 0), colspan=1);
bar(train,'Health', 'Pet Health', 'Pet Health');
plt.subplot2grid((1,3), (0, 1), colspan=1);
prop_stacked_bar(train, 'Health', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Health', 'Adoption Speed Rate wrt Pet Health');
plt.subplot2grid((1,3), (0, 2), colspan=1);
bar(test,'Health', 'Pet Health (Test Set)', 'Pet Health (Test Set)');
fig.canvas.set_window_title('Overall Health')

fig = plt.figure(figsize=(14, 6));
plt.subplot2grid((2, 3), (0, 0), colspan=1)
basic_plot(train,"Age", "Count", "Pet Months", "Pet Months")
train['AgeAges'] = train['Age'].apply(lambda x: math.floor(x/12));
plt.subplot2grid((2, 3), (0, 1), colspan=1);
bar(train,'AgeAges', 'Pet Age', 'Pet Age', showPercentages=False, showCount=True, size="xx-small");
plt.subplot2grid((2,3), (0, 2), colspan=1);
prop_stacked_bar(train, 'AgeAges', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Age', 'Adoption Speed Rate wrt Pet Age');
plt.subplot2grid((2,3), (1, 0), colspan=1);
prop_stacked_bar(train, 'AgeCat', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Age Category', 'Adoption Speed Rate wrt Pet Age Category');
plt.subplot2grid((2,3), (1, 1), colspan=1);
prop_stacked_bar(train[train['Type'] == "cats"], 'AgeCat', 'AdoptionSpeed', 'Adoption Speed Rate', 'Cat Age Category', 'Adoption Speed Rate wrt Cat Age Category');
plt.subplot2grid((2,3), (1, 2), colspan=1);
prop_stacked_bar(train[train['Type'] == "dogs"], 'AgeCat', 'AdoptionSpeed', 'Adoption Speed Rate', 'Dog Age Category', 'Adoption Speed Rate wrt Dog Age Category');
plt.tight_layout()
fig.canvas.set_window_title('Age')

fig = plt.figure(figsize=(14, 6));
plt.subplot2grid((3,4), (0, 0), colspan=1);
bar(train,'VideoAmt', 'Num of Pet Videos', 'Num of Pet Videos',showPercentages=False, showCount=True);
plt.subplot2grid((3,4), (0, 1), colspan=1);
prop_stacked_bar(train, 'VideoAmt', 'AdoptionSpeed', 'Adoption Speed Rate', 'Num of Pet Videos', 'Adoption Speed Rate wrt NoPV');
plt.subplot2grid((3,4), (0, 2), colspan=1);
prop_stacked_bar(train, 'HasVideos', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Has Video/s', 'Adoption Speed Rate wrt PHV');
plt.subplot2grid((3,4), (0, 3), colspan=1);
prop_stacked_bar(train, 'HasPhotos', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Has Photo/s', 'Adoption Speed Rate wrt PHP');
plt.subplot2grid((3,4), (1, 0), colspan=4);
bar(train,'PhotoAmt', 'Num of Pet Photos', 'Num of Pet Photos',showPercentages=False, showCount=True);
plt.subplot2grid((3,4), (2, 0), colspan=4);
prop_stacked_bar(train, 'PhotoAmt', 'AdoptionSpeed', 'Adoption Speed Rate', 'Num of Pet Photos', 'Adoption Speed Rate wrt NoPP');
plt.tight_layout()
fig.canvas.set_window_title('Videos, Photos')

fig = plt.figure(figsize=(14, 6));
plt.subplot2grid((2,2), (0, 0), colspan=1);
bar(train,'FurLength', 'Pet Fur Length', 'Pet Fur Length');
plt.subplot2grid((2,2), (0, 1), colspan=1);
prop_stacked_bar(train, 'FurLength', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Fur Length', 'Adoption Speed Rate wrt Pet Fur Length');
plt.subplot2grid((2,2), (1, 0), colspan=1);
bar(train,'MaturitySize', 'Pet Maturity Size', 'Pet Maturity Size');
plt.subplot2grid((2,2), (1, 1), colspan=1);
prop_stacked_bar(train, 'MaturitySize', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Maturity Size', 'Adoption Speed Rate wrt Pet Maturity Size');
plt.tight_layout()
fig.canvas.set_window_title('Fur Length, Maturity Size')

fig = plt.figure(figsize=(14, 6));
plt.subplot2grid((3,1), (0, 0), colspan=1);
bar(train,'Quantity', 'Pet Quantity', 'Pet Quantity',showPercentages=False,showCount=True);
plt.subplot2grid((3,1), (1, 0), colspan=1);
prop_stacked_bar(train, 'Quantity', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Quantity', 'Adoption Speed Rate wrt Pet Quantity');
train.loc[train.Quantity > 6, "Quantity"] = 6
plt.subplot2grid((3,1), (2, 0), colspan=1);
prop_stacked_bar(train, 'Quantity', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet Quantity', 'Adoption Speed Rate wrt Pet Quantity');
plt.tight_layout()
fig.canvas.set_window_title('Quantity')

fig = plt.figure(figsize=(14, 6));
plt.subplot2grid((2,4), (0, 0), colspan=1);
wordcloud_plot(train[train['Type'] == "cats"]['Breed1'].fillna('').value_counts(),'Most Popular Breed1 Cats',freq=True)
plt.subplot2grid((2,4), (0, 1), colspan=1);
wordcloud_plot(train[train['Type'] == "dogs"]['Breed1'].fillna('').value_counts(),'Most Popular Breed1 Dogs',freq=True)
plt.subplot2grid((2,4), (0, 2), colspan=1);
wordcloud_plot(train[train['Type'] == "cats"]['Breed2'].fillna('').value_counts(),'Most Popular Breed2 Cats',freq=True)
plt.subplot2grid((2,4), (0, 3), colspan=1);
wordcloud_plot(train[train['Type'] == "dogs"]['Breed2'].fillna('').value_counts(),'Most Popular Breed2 Dogs',freq=True)
plt.subplot2grid((2,4), (1, 0), colspan=1);
bar(train,'PureBreed', 'Pure Breed', 'Pure Breed',rotation=45);
plt.subplot2grid((2,4), (1, 1), colspan=1);
prop_stacked_bar(train, 'PureBreed', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pure Breed', 'Adoption Speed Rate wrt Pure Breed',rotate=True);
plt.subplot2grid((2,4), (1, 2), colspan=1);
prop_stacked_bar(train[train['Type'] == "cats"], 'PureBreed', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pure Breed Cats', 'Adoption Speed Rate wrt Pure Breed Cats',rotate=True);
plt.subplot2grid((2,4), (1, 3), colspan=1);
prop_stacked_bar(train[train['Type'] == "dogs"], 'PureBreed', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pure Breed Dogs', 'Adoption Speed Rate wrt Pure Breed Dogs',rotate=True);
plt.tight_layout()
fig.canvas.set_window_title('Breed')

fig = plt.figure(figsize=(14, 6));
plt.subplot2grid((2,2), (0, 0), colspan=1);
bar(train,'Color1', 'Most Popular Colors', 'Most Popular Colors');
plt.subplot2grid((2,2), (0, 1), colspan=1);
prop_stacked_bar(train, 'Color1', 'AdoptionSpeed', 'Adoption Speed Rate', 'Most Popular Colors', 'Adoption Speed Rate wrt Most Popular Colors');
plt.subplot2grid((2,2), (1, 0), colspan=1);
bar(train,'Colorful', 'Pet colorfulness', 'Pet colorfulness');
plt.subplot2grid((2,2), (1, 1), colspan=1);
prop_stacked_bar(train, 'Colorful', 'AdoptionSpeed', 'Adoption Speed Rate', 'Pet colorfulness', 'Adoption Speed Rate wrt Pet colorfulness');
plt.tight_layout()
fig.canvas.set_window_title('Color')

fig = plt.figure(figsize=(14, 6));
plt.subplot2grid((1,2), (0, 0), colspan=1);
basic_plot(train,"Fee", "Count", "Fee", "Fee")
plt.subplot2grid((1,2), (0, 1), colspan=1);
prop_stacked_bar(train, 'FeeCat', 'AdoptionSpeed', 'Adoption Speed Rate', 'Fee Cat', 'Adoption Speed Rate wrt Fee Cat');
plt.tight_layout()
fig.canvas.set_window_title('Fee')

fig = plt.figure(figsize=(14, 6));
plt.subplot2grid((1,3), (0, 0), colspan=1);
basic_plot(train,"DescLength", "Count", "Description Length", "Description Length")
plt.subplot2grid((1,3), (0, 1), colspan=1);
bar(train,'HasDesc', 'Has Description', 'Has Description');
plt.subplot2grid((1,3), (0, 2), colspan=1);
prop_stacked_bar(train, 'HasDesc', 'AdoptionSpeed', 'Adoption Speed Rate', 'Has Description', 'Adoption Speed Rate wrt Has Description');
plt.tight_layout()
fig.canvas.set_window_title('Description')

rescuers = train.loc[train['RescuerID'].isin(list(train.RescuerID.value_counts().index[1:15]))]
fig = plt.figure(figsize=(14, 6));
plt.subplot2grid((2,1), (0, 0), colspan=1);
bar(rescuers,'RescuerID', 'Rescuer', 'Rescuer',showPercentages=False,showCount=True,endcodeLabels=False);
plt.subplot2grid((2,1), (1, 0), colspan=1);
prop_stacked_bar(rescuers, 'RescuerID', 'AdoptionSpeed', 'Adoption Speed Rate', 'Rescuer', 'Adoption Speed Rate wrt Rescuer',endcodeLabels=False);
plt.tight_layout()
fig.canvas.set_window_title('Rescuers')

plt.show();