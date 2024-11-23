from mne.time_frequency import psd_array_multitaper
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna import Trial
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
# 데이터 불러오기
arr_1 = np.load('train.npy', allow_pickle=True)
arr_2 = np.load('test.npy', allow_pickle=True)
dic = arr_1.tolist()
test = arr_2.tolist()
x = dic["input"].reshape(1080, 256, -1)
y = dic["label"]
test = test["input"].reshape(120, 256, -1)
# 학습데이터와 테스트데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=777, stratify=y)
x = X_train
y = y_train
# 주파수 대역 설정하기
freq_ranges = {'a': (0.5, 5), 'b': (5, 10), 'c': (10, 15), 'd': (15, 28), 'e': (31, 43)}
# PSD 계산 함수
def compute_psd(data, fs):
    psd_features = []
    for freq_band, freq_range in freq_ranges.items():
        freq_min, freq_max = freq_range
        psd, freqs = psd_array_multitaper(data.T, sfreq=fs, fmin=freq_min, fmax=freq_max)
        psd_feature = np.median(psd)
        psd_features.append(psd_feature)
    return psd_features
# 샘플링 주파수
fs = 256
# 각 데이터에 대해 PSD 계산
psd_features_all_train = []
for sample in x:
    psd_features_sample = compute_psd(sample, fs)
    psd_features_all_train.append(psd_features_sample)
psd_features_all_test = []
for sample in test:
    psd_features_sample = compute_psd(sample, fs)
    psd_features_all_test.append(psd_features_sample)
psd_features_all_X_test = []
for sample in X_test:
    psd_features_sample = compute_psd(sample, fs)
    psd_features_all_X_test.append(psd_features_sample)
# PSD를 데이터 프레임으로 변환
psd_train_df = pd.DataFrame(psd_features_all_train, columns=['a', 'b', 'c', 'd', 'e'])
psd_test_df = pd.DataFrame(psd_features_all_test, columns=['a', 'b', 'c', 'd', 'e'])
psd_X_test_df = pd.DataFrame(psd_features_all_X_test, columns=['a', 'b', 'c', 'd', 'e'])
# 데이터 정규화
scaler = MinMaxScaler()
x = scaler.fit_transform(psd_train_df)
X_test = scaler.transform(psd_X_test_df)
test_real = scaler.transform(psd_test_df)
# 옵튜나로 하이퍼 파라미터 최적화
def objective(trial: Trial, model, x, y, eval_metric):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float('learning_rate', 0.0001, 0.99),
        'n_estimators': trial.suggest_int("n_estimators", 1000, 10000, step=100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 15),
        "gamma": trial.suggest_float("gamma", 0.1, 1.0, log=True),
    }
    results = dict()
    fold = StratifiedKFold(n_splits=8, shuffle=True, random_state=777)
    for i, (train_index, valid_index) in tqdm(enumerate(fold.split(x, y))):
        x_train, y_train = x[train_index], y[train_index]
        x_valid, y_valid = x[valid_index], y[valid_index]
        k_model = model(**params, early_stopping_rounds=15)
        eval_set = [(x_valid, y_valid)]
        k_model.fit(x_train, y_train, eval_set=eval_set)
        pred_valid_labels = k_model.predict(x_valid)
        error = eval_metric(y_valid, pred_valid_labels)
        accuracy_valid = accuracy_score(y_valid, pred_valid_labels)
        pred_train_labels = k_model.predict(x_train)
        accuracy_train = accuracy_score(y_train, pred_train_labels)
        # 결과 저장
        results[i] = {'model': k_model, 'error': error}
        print(f"sub: {accuracy_train - accuracy_valid:.4f}, train_accuracy: {accuracy_train:.4f}, valid_accuracy: {accuracy_valid:.4f}")
    # 모든 Fold의 평가 결과의 평균을 반환
    errors = [v['error'] for k, v in results.items()]
    return np.array(errors).mean()
# 최소화 방향으로 하이퍼 파라미터 학습을 위한 스터디 객체생성함
study = optuna.create_study(direction='minimize', study_name='XGboost')
# 하이퍼파라미터 최적화 시작
study.optimize(lambda trial: objective(trial, XGBClassifier, x, y, eval_metric=log_loss), n_trials=100)
print("---Best trial:")
best_trial = study.best_trial
print("  ---Value: ", best_trial.value)
print("  ---Params: ")
for key, value in best_trial.params.items():
    print(f"    '{key}': {value}")
mean_cross_val_loss = np.mean([v['error'] for v in best_trial.user_attrs.values()])
# 오차행렬 시각화 함수
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
# 모델을 평가하고 오차행렬과 ROC곡선 그리는 함수
def evaluate_model_with_test_data(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=[0, 1], title='Confusion Matrix')
    plt.show()
    # 모델 평가 척도
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_test, y_pred)
    # ROC-AUC점수 계산후 ROC곡선 그리기
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"roc_auc: {roc_auc:.2f}")
# 모델 학습 및 평가
best_params = study.best_params
model = XGBClassifier(**study.best_params)
model.fit(x, y)
evaluate_model_with_test_data(model, X_test, y_test)
# test데이터로 예측 뽑아내고 제출용 csv파일 만들어냄
predict_test = model.predict(test_real)
predict_test = predict_test.astype(np.int32)
df = pd.DataFrame(predict_test, columns=["TARGET"])
df.to_csv('output.csv', index_label="ID")