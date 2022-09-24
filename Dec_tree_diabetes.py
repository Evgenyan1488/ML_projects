import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 20)


csv_input = pd.read_csv("CSV/Diabetes/diabetes.csv")

y = csv_input['Outcome']
x = csv_input.loc[:, 'Pregnancies':'Age']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)


clf = DecisionTreeClassifier(max_depth=4)


clf = clf.fit(x_train, y_train)

forestclf = RandomForestClassifier()


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
max_features = ['log2', 'sqrt']
max_depth = [int(x) for x in np.linspace(start = 1, stop = 15, num = 15)]
min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
bootstrap = [True, False]
param_dist = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rs = RandomizedSearchCV(forestclf,
                        param_dist,
                        n_iter = 100,
                        cv = 3,
                        verbose = 1,
                        n_jobs=-1,
                        random_state=0)
rs.fit(x_train, y_train)
forestclf = rs.best_estimator_

rs_df = pd.DataFrame(rs.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
rs_df = rs_df.drop([
            'mean_fit_time',
            'std_fit_time',
            'mean_score_time',
            'std_score_time',
            'params',
            'split0_test_score',
            'split1_test_score',
            'split2_test_score',
            'std_test_score'],
            axis=1)

# fig, axs = plt.subplots(ncols=3, nrows=2)
# sns.set(style="whitegrid", color_codes=True, font_scale = 2)
# fig.set_size_inches(30,25)
# sns.barplot(x='param_n_estimators', y='mean_test_score', data=rs_df, ax=axs[0,0], color='lightgrey')
# axs[0,0].set_ylim([.83,.93])
# axs[0,0].set_title(label = 'n_estimators', size=30, weight='bold')
# sns.barplot(x='param_min_samples_split', y='mean_test_score', data=rs_df, ax=axs[0,1], color='coral')
# axs[0,1].set_ylim([.85,.93])
# axs[0,1].set_title(label = 'min_samples_split', size=30, weight='bold')
# sns.barplot(x='param_min_samples_leaf', y='mean_test_score', data=rs_df, ax=axs[0,2], color='lightgreen')
# axs[0,2].set_ylim([.80,.93])
# axs[0,2].set_title(label = 'min_samples_leaf', size=30, weight='bold')
# sns.barplot(x='param_max_features', y='mean_test_score', data=rs_df, ax=axs[1,0], color='wheat')
# axs[1,0].set_ylim([.88,.92])
# axs[1,0].set_title(label = 'max_features', size=30, weight='bold')
# sns.barplot(x='param_max_depth', y='mean_test_score', data=rs_df, ax=axs[1,1], color='lightpink')
# axs[1,1].set_ylim([.80,.93])
# axs[1,1].set_title(label = 'max_depth', size=30, weight='bold')
# sns.barplot(x='param_bootstrap',y='mean_test_score', data=rs_df, ax=axs[1,2], color='skyblue')
# axs[1,2].set_ylim([.88,.92])
# axs[1,2].set_title(label = 'bootstrap', size=30, weight='bold')
# plt.show()
print(rs.best_params_)

#clf2 = RandomForestClassifier(rs.best_params_)
#clf2 = clf2.fit(x_train, y_train)


#y_pred = clf.predict(x_test)

print("Train accuracy:", clf.score(x_train, y_train))
print("Test accuracy:", clf.score(x_test, y_test))
print("F Train accuracy:", forestclf.score(x_train, y_train))
print("F Test accuracy:", forestclf.score(x_test, y_test))

importances = forestclf.feature_importances_
indices = np.argsort(importances)[::-1]

col = x.columns

ar_f=[]
for f, idx in enumerate(indices):
    ar_f.append([round(importances[idx],4), col[idx]])
print("Значимость признака:")
ar_f.sort(reverse=True)
print(ar_f)