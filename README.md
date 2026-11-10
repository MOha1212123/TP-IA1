# TP-IA1

# M115 Lab 1: Phishing URL Detection
# Solution code for Exercises 1-10.
#
# HOW TO USE: copy ONLY the lines between "START CODE HERE" and "END CODE HERE"
# of each exercise into the matching cell of your notebook, replacing the `None`
# placeholders. Everything else in each cell stays as it is in the template.


# =====================================================================
# EXERCISE 1: Data Quality & SOC Target Alignment
# =====================================================================
### START CODE HERE ###
df['IsPhishing'] = 1 - df['label']          # raw label: 1 = Legitimate, 0 = Phishing -> flip it
phishing_ratio = df['IsPhishing'].mean()
### END CODE HERE ###


# =====================================================================
# EXERCISE 2: Binary Security Indicators per Class
# =====================================================================
### START CODE HERE ###
binary_stats = df.groupby('IsPhishing')[binary_cols].mean() * 100   # rows = class (0/1), cols = indicators
### END CODE HERE ###


# =====================================================================
# EXERCISE 3: Constructing X, y and Correlations
# =====================================================================
### START CODE HERE ###
X = df.drop(columns=cols_to_drop, errors='ignore')
y = df['IsPhishing']
correlations = X.corrwith(y).dropna().sort_values()   # ascending: negatives first, positives last
### END CODE HERE ###


# =====================================================================
# EXERCISE 4: Stratified Split + Leakage-Free Standardization
# =====================================================================
### START CODE HERE ###
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED
)
scaler = StandardScaler().fit(X_train)             # fit on TRAIN only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)           # transform only, no refit
### END CODE HERE ###


# =====================================================================
# EXERCISE 5: Training & Comparing 5 Logistic Regression Configurations
# (inside the `for name, model in configurations.items():` loop)
# =====================================================================
### START CODE HERE ###
model.fit(X_train_scaled, y_train)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob >= 0.50).astype(int)
w = model.coef_.ravel()
n_zero_weights = int(np.sum(np.abs(w) < 1e-4))
### END CODE HERE ###


# =====================================================================
# EXERCISE 6: L1 Regularization Path
# (inside the `for c_val in C_values:` loop)
# =====================================================================
### START CODE HERE ###
m_l1 = LogisticRegression(penalty='l1', C=c_val, solver='liblinear', max_iter=1000, random_state=SEED)
m_l1_fit = m_l1.fit(X_train_scaled, y_train)
n_active = int(np.sum(np.abs(m_l1_fit.coef_.ravel()) >= 1e-4))
### END CODE HERE ###


# =====================================================================
# EXERCISE 7: Log-Odds (w_j) and Odds Ratios (exp(w_j))
# =====================================================================
### START CODE HERE ###
w_l2 = model_l2.coef_.ravel()
w_l1 = model_l1.coef_.ravel()
weights_df = pd.DataFrame({
    'Feature': X.columns,
    'w_j (L2, M3)': w_l2,
    'Odds Ratio exp(w_j)': np.exp(w_l2),
    'w_j (L1, M4)': w_l1,
}).sort_values('w_j (L2, M3)', ascending=False).set_index('Feature')
### END CODE HERE ###


# =====================================================================
# EXERCISE 8: SHAP Values (LinearExplainer)
# =====================================================================
### START CODE HERE ###
explainer = shap.LinearExplainer(model_l2, X_train_scaled)   # background = training distribution
shap_values = explainer(X_test_sample)
### END CODE HERE ###


# =====================================================================
# EXERCISE 9: Locate TP, TN and FP cases in the 1,000-URL sample
# =====================================================================
### START CODE HERE ###
tp_idx = int(np.where((y_test_sample == 1) & (y_pred_sample == 1))[0][0])
tn_idx = int(np.where((y_test_sample == 0) & (y_pred_sample == 0))[0][0])
fp_candidates = np.where((y_test_sample == 0) & (y_pred_sample == 1))[0]
### END CODE HERE ###


# =====================================================================
# EXERCISE 10: Operational Decision Thresholds
# (inside the `for tau in thresholds:` loop)
# =====================================================================
### START CODE HERE ###
y_pred_tau = (y_prob_full >= tau).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tau).ravel()
### END CODE HERE ###
