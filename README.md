9. Your script illustrates:

Training Decision Trees using different splitting criteria

Visualizing each tree

Demonstrating Decision Tree instability using bootstrap resampling

1. Importing Libraries & Loading the Iris Dataset
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names


Loads the classic Iris flower classification dataset (4 features, 3 classes).

X: input features

y: true class labels

feature_names: names of the features for tree plotting

2. Train–Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


Splits 30% of data for testing.

stratify=y ensures all classes remain balanced across train/test.

3. Training Models with Different Criteria
criteria = ["gini", "entropy"]


You loop over two impurity measures:

Gini

Default metric

Measures how often a randomly chosen element would be misclassified

Entropy (Information Gain)

Based on information theory

Measures reduction in uncertainty

Both criteria typically give similar results, but entropy is more computationally expensive.

Training each Decision Tree
dt = DecisionTreeClassifier(
        criterion=crit,
        max_depth=3,
        min_samples_split=4,
        random_state=42
)


Important hyperparameters:

max_depth=3: forces a shallow tree → reduces overfitting

min_samples_split=4: splits only if at least 4 samples present

random_state=42: ensures reproducible tree structure

Evaluating & Printing Results
y_pred = dt.predict(X_test)
y_proba = dt.predict_proba(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Feature Importances:", dt.feature_importances_)


Calculates accuracy on the test set

Prints feature importance values (how much each feature contributes)

Plotting the Decision Tree
plot_tree(dt, feature_names=feature_names,
          class_names=data.target_names,
          filled=True)


Visualizes the tree structure

Node colors show the dominant class

Text boxes show:

splitting feature and threshold

Gini/entropy impurity

sample counts

4. Instability Demonstration (Bootstrap)

Decision Trees are known to be unstable:
Small changes in training data can lead to very different trees.

This instability motivates ensembles like Random Forests.

Bootstrap Sampling
idx = np.random.choice(len(X_train), size=len(X_train), replace=True)


Creates a new dataset by sampling with replacement

Some samples appear multiple times, some not at all

Train a tree on each bootstrap sample
dt_bs = DecisionTreeClassifier(max_depth=3)
dt_bs.fit(X_b, y_b)
importances.append(dt_bs.feature_importances_)


Using max_depth=3 exaggerates the instability

Collects each bootstrap tree's feature importances

5. Visualizing Instability via Boxplot
sns.boxplot(data=importances)
plt.xticks(ticks=range(len(feature_names)), labels=feature_names)
plt.title("Decision Tree Instability (Bootstrap Feature Importances)")


This boxplot shows:

Variation in feature importance across bootstrap samples

Wide boxes = high instability

If different trees pick different splits early, feature importance varies a lot

This is why:

Decision Trees alone can be unreliable

Random Forests average many bootstrapped trees → much more stable

✅ Summary: What the Script Demonstrates
✔ Trains decision trees with Gini vs. Entropy
✔ Prints accuracy and feature importance
✔ Visualizes each tree
✔ Demonstrates model instability using bootstrapping
✔ Produces a boxplot showing the variance in feature importances

This is a great educational demonstration of both how Decision Trees work and why ensemble methods exist.




10. 🌳 1. Single Decision Tree (Baseline)
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)


A single tree is fast and interpretable but unstable.

Small training changes → big structural changes in the tree.

This instability usually leads to higher variance and weaker accuracy.

This serves as your baseline for comparison.

👜 2. Bagging (Bootstrap Aggregating)
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=0
)


How it works:

Creates many bootstrap samples of the training data.

Trains 50 independent trees.

Final prediction = majority vote across all trees.

✔ Strengths:

Reduces variance

Improves stability

Works especially well for unstable models like decision trees

Bagging = "Many noisy models averaged together → smooth stable predictor."

🌲 3. Random Forest (Bagging + Feature Randomness)
rf = RandomForestClassifier(
    n_estimators=200,
    oob_score=True,
    bootstrap=True,
    random_state=0
)


Random Forest = Bagging plus:

At each split, each tree only sees a subset of random features.

This reduces the correlation between trees → stronger ensemble.

✔ Benefits over Bagging:

Lower variance

Better generalization

Built-in feature importance

OOB score acts like automatic cross-validation

print("Random Forest OOB Score:", rf.oob_score_)


OOB score = how well the forest predicts samples not included in each tree’s bootstrap sample.

⚡ 4. AdaBoost (Adaptive Boosting)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.8,
    random_state=0
)


Boosting ≠ Bagging.

Boosting trains a sequence of trees:

Each new tree focuses on misclassified samples from the previous trees.

Using max_depth=1 makes each tree a decision stump (weak learner).

The final model is a weighted sum of many weak learners.

✔ Strengths:

Excellent performance on simple datasets

Reduces both bias and variance

Often beats random forests on low-noise datasets

🧠 5. Stacking (Meta-Learning)
estimators = [
    ("dt", DecisionTreeClassifier(max_depth=3)),
    ("svm", SVC(probability=True))
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)


Stacking = “models of models.”

Train multiple base models (tree + SVM here).

Use their predictions as features.

Train a meta-classifier (here Logistic Regression) on those predictions.

✔ Strength:

Combines multiple heterogeneous models

Often gives the best accuracy

Meta-learner learns how to optimally blend base models

📊 6. Summary Table
print(f"Decision Tree : {accuracy_score(y_test, dt_pred):.3f}")
print(f"Bagging       : {accuracy_score(y_test, bag_pred):.3f}")
print(f"RandomForest  : {accuracy_score(y_test, rf_pred):.3f}")
print(f"AdaBoost      : {accuracy_score(y_test, ada_pred):.3f}")
print(f"Stacking      : {accuracy_score(y_test, stack_pred):.3f}")


On the Iris dataset, typical results are:

Model	Expected Behavior
Decision Tree	Lowest accuracy (highest variance)
Bagging	Better than single tree
Random Forest	Usually the best or tied-best
AdaBoost	Often very strong on simple datasets
Stacking	Can be best if base models are diverse
🎯 In Simple Terms

Decision Tree: Simple but unstable

Bagging: Stabilizes the tree by averaging many trees

Random Forest: Bagging + random features → even better

AdaBoost: Trains trees sequentially to fix previous mistakes

Stacking: Combines different models using a meta-model


11. This script demonstrates how dimensionality-reduction techniques affect visualization and k-NN classification on the Digits dataset (8×8 images → 64 features).

1. Loading the Data
digits = load_digits()
X = digits.data
y = digits.target


Each sample is an 8×8 grayscale image.

The 64 pixel intensities become a 64-dimensional feature vector.

y contains labels 0–9.

2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


80% for training, 20% for testing.

3. Baseline k-NN Accuracy (64 dimensions)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, knn.predict(X_test))


k-NN works well in moderate dimensions.

Serves as a baseline before reducing the data.

4. Correct Evaluation Function

Your original version was wrong because it re-split Y while splitting reduced X.

The correct version:

def evaluate_reduction(reducer, name):
    X_train_r = reducer.fit_transform(X_train)
    X_test_r = reducer.transform(X_test)
    ...


Why this is correct:

Dimensionality reduction must be fitted ONLY on the training set → proper ML practice.

The test set should only be transformed, not learned from.

5. Principal Component Analysis (PCA)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


Linear dimensionality reduction.

Finds directions of maximal variance.

Often produces decent clusters.

Not good for nonlinear structure.

6. Kernel PCA (RBF Kernel)
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.01)


Nonlinear version of PCA.

Uses the RBF kernel to capture curved manifolds.

Typically produces a more separated embedding in 2D.

7. Locally Linear Embedding (LLE)
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)


Nonlinear manifold learning.

Preserves local structure.

Good at revealing hidden geometry (e.g., Swiss roll datasets).

Can be unstable depending on parameters.

8. Visualizing PCA, Kernel PCA, and LLE
scatter = ax.scatter(...)


Colors represent digit classes.

The last plot gets a colorbar.

9. Expected Behavior
PCA

Clusters appear but often overlap.

Good global structure.

Kernel PCA

Curved, nonlinear mapping.

Better separation for many digits.

LLE

Strong local structure.

Can produce complex, sometimes twisted embeddings.

10. Expected Accuracy Results

Typically:

Method	Accuracy
No reduction	best (high-dimensional kNN works well)
PCA 2D	moderate
Kernel PCA 2D	better
LLE 2D	varies; sometimes unstable

Why accuracy drops:
You reduce 64 dimensions → 2 dimensions → massive information loss.
