import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

np.random.seed(53)

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))

# 1000サンプルの正常値学習データを用意した
X = 0.4 * np.random.randn(250, 2)
X_train = np.r_[X+3, X+1, X-1, X-3]

# 10％（100サンプル）にあたる異常学習データを混ぜた
X_train = np.r_[X_train, np.random.uniform(low=-4, high=4, size=(100, 2))]

# 200サンプルの正常値テストデータを用意した
X = 0.4 * np.random.randn(50, 2)
X_test = np.r_[X+3, X+1, X-1, X-3]

# 20サンプルの異常値テストデータを用意した
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

error_delete = 0
normal_delete = 0
# 10%くらいうまく取れないことがあると仮定している
for i in range(100):
    clf = svm.OneClassSVM(kernel="rbf", gamma=(0.1), nu=(0.1))
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    n_error_train = y_pred_train[y_pred_train == -1].size

    # 外れ値を外す処理
    worst = np.argmin(clf.decision_function(X_train))

    # 外した外れ値が後から追加したエラーか確認する
    if worst >= 1000:
        error_delete += 1
    else:
        normal_delete += 1
    X_train = np.delete(X_train, worst, 0)

    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("Novelty Detection [gamma:" +str(0.1)+ ",nu:"+str(0.1)+"]")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

    s = 40
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                     edgecolors='k')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                    edgecolors='k')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([a.collections[0], b1, b2, c],
               ["learned frontier", "training observations",
                "new regular observations", "new abnormal observations"],
               loc="upper left",
               prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlabel(
        "error train: %d/%d ; errors novel regular: %d/400 ; "
        "errors novel abnormal: %d/40"
        % (n_error_train, X_train.size, n_error_test, n_error_outliers))
    #plt.show()
    plt.savefig('img/ocsvm_4_'+str(i)+'.png')
    plt.close()

print("error_delete = " + str(error_delete))
print("normal_delete = " + str(normal_delete))
