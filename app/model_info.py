# model_info.py
# Descriptive metadata for each regression model used in the pipeline.
# Each entry contains: full name, description, historical background,
# governing equation, and hyperparameter descriptions.

MODEL_INFO = {
    "OLS": {
        "name": "Ordinary Least Squares",
        "description": "Models the linear relationship between inputs and output by minimising the sum of squared residuals.",
        "history": "Independently developed by Carl Friedrich Gauss and Adrien-Marie Legendre in the early 1800s.",
        "equation": "y = β₀ + Σᵢ βᵢ xᵢ",
        "parameters": {
            "fit_intercept": "Whether to include the intercept term β₀.",
            "normalize": "Standardises inputs to zero mean and unit variance before fitting."
        }
    },
    "XGBoost": {
        "name": "eXtreme Gradient Boosting",
        "description": "Tree-based gradient boosting algorithm optimised for speed and predictive performance.",
        "history": "Published by Tianqi Chen and Carlos Guestrin in 2016.",
        "equation": "Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)",
        "parameters": {
            "n_estimators": "Total number of boosting trees.",
            "max_depth": "Maximum depth of each tree.",
            "learning_rate": "Step size shrinkage per iteration (η).",
            "subsample": "Fraction of training samples used per tree.",
            "colsample_bytree": "Fraction of features sampled per tree.",
            "reg_alpha": "L1 regularisation coefficient.",
            "reg_lambda": "L2 regularisation coefficient."
        }
    },
    "LightGBM": {
        "name": "Light Gradient Boosting Machine",
        "description": "Histogram-based gradient boosting with leaf-wise growth, designed for large datasets.",
        "history": "Introduced by Microsoft in 2017.",
        "equation": "Leaf-wise growth: Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)",
        "parameters": {
            "n_estimators": "Total number of boosting trees.",
            "max_depth": "Maximum tree depth.",
            "learning_rate": "Step size per iteration (η).",
            "num_leaves": "Maximum number of leaves per tree."
        }
    },
    "Ridge": {
        "name": "Ridge Regression (L2 Regularisation)",
        "description": "Constrains coefficient magnitudes with an L2 penalty to prevent overfitting.",
        "history": "Developed by A. E. Hoerl and R. W. Kennard in the 1970s.",
        "equation": "minimise ||y – Xβ||² + α ||β||²",
        "parameters": {
            "alpha": "L2 penalty strength.",
            "solver": "Optimisation algorithm (auto, svd, lsqr)."
        }
    },
    "Lasso": {
        "name": "Lasso Regression (L1 Regularisation)",
        "description": "Drives some coefficients to exactly zero, performing implicit feature selection.",
        "history": "Proposed by Robert Tibshirani in 1996.",
        "equation": "minimise ||y – Xβ||² + α ||β||₁",
        "parameters": {
            "alpha": "L1 penalty strength.",
            "selection": "Coefficient update order (cyclic or random)."
        }
    },
    "Elastic": {
        "name": "Elastic Net",
        "description": "Combines L1 and L2 penalties to achieve both feature selection and coefficient shrinkage.",
        "history": "Introduced by H. Zou and T. Hastie in 2005.",
        "equation": "minimise ||y – Xβ||² + α [ρ ||β||₁ + (1–ρ) ||β||²]",
        "parameters": {
            "alpha": "Overall regularisation strength.",
            "l1_ratio": "Fraction of L1 penalty (ρ)."
        }
    },
    "Bayesian": {
        "name": "Bayesian Ridge Regression",
        "description": "Places a Bayesian prior over regression coefficients, inferring their distribution from data.",
        "history": "Builds on work by David J. C. MacKay and Christopher M. Bishop.",
        "equation": "Coefficient estimation via posterior maximisation.",
        "parameters": {
            "alpha_1": "Shape parameter α₁ of the coefficient prior.",
            "alpha_2": "Rate parameter α₂ of the coefficient prior.",
            "lambda_1": "Shape parameter λ₁ of the noise prior.",
            "lambda_2": "Rate parameter λ₂ of the noise prior."
        }
    },
    "ARD": {
        "name": "Automatic Relevance Determination Regression",
        "description": "Learns a separate penalty parameter per feature, shrinking irrelevant ones to zero.",
        "history": "Rooted in Bayesian sparse modelling work by Neal and MacKay in the 1990s.",
        "equation": "Individual αᵢ estimation via posterior maximisation.",
        "parameters": {
            "max_iter": "Maximum number of iterations.",
            "alpha_1": "Shape parameter α₁ per feature.",
            "alpha_2": "Rate parameter α₂ per feature.",
            "lambda_1": "Shape parameter λ₁ of the noise prior.",
            "lambda_2": "Rate parameter λ₂ of the noise prior."
        }
    },
    "Huber": {
        "name": "Huber Regressor",
        "description": "Uses the Huber loss function to reduce sensitivity to outliers.",
        "history": "Formulated by Peter J. Huber in 1964.",
        "equation": "Huber loss: squared for |error| ≤ ε, absolute otherwise.",
        "parameters": {
            "epsilon": "Threshold ε separating quadratic from linear loss.",
            "alpha": "L2 regularisation strength."
        }
    },
    "RANSAC": {
        "name": "RANSAC Regressor",
        "description": "Fits a model on random subsets of data and discards outliers iteratively.",
        "history": "Proposed by Fischler and Bolles in 1981.",
        "equation": "— (iterative random subset consensus)",
        "parameters": {
            "max_trials": "Number of random subsets drawn.",
            "residual_threshold": "Maximum residual to be considered an inlier."
        }
    },
    "TheilSen": {
        "name": "Theil–Sen Estimator",
        "description": "Computes the median slope over all point pairs, yielding a robust linear fit.",
        "history": "Based on Henri Theil (1950) and Pranab Sen (1968).",
        "equation": "median((y_j – y_i) / (x_j – x_i))",
        "parameters": {
            "n_subsamples": "Number of subsamples drawn.",
            "max_subpopulation": "Maximum subpopulation size for slope computation."
        }
    },
    "PLS": {
        "name": "Partial Least Squares Regression",
        "description": "Combines dimensionality reduction via latent variables with regression.",
        "history": "Developed by Sven Wold in the 1970s.",
        "equation": "X = T Pᵀ + E;  y = T q + f",
        "parameters": {
            "n_components": "Number of latent components."
        }
    },
    "MLP": {
        "name": "Multi-Layer Perceptron Regressor",
        "description": "Fully-connected neural network for non-linear regression.",
        "history": "Extends Frank Rosenblatt's 1958 perceptron; multi-layer networks became widespread in the 1980s.",
        "equation": "y = f(x; W, b)",
        "parameters": {
            "hidden_layer_sizes": "Tuple specifying the number of units in each hidden layer.",
            "activation": "Activation function (relu, tanh, etc.).",
            "alpha": "L2 regularisation coefficient.",
            "learning_rate": "Learning rate schedule (constant or adaptive)."
        }
    },
    "SVM": {
        "name": "Support Vector Regression",
        "description": "Applies the support vector machine principle with an ε-insensitive loss function.",
        "history": "Derived from Vladimir Vapnik and Alexey Chervonenkis's work circa 1995.",
        "equation": "minimise ½||w||² + C Σ L_ε(yᵢ – wᵀφ(xᵢ))",
        "parameters": {
            "kernel": "Kernel function (rbf, linear).",
            "C": "Regularisation parameter.",
            "gamma": "Kernel coefficient for RBF.",
            "epsilon": "Width of the ε-insensitive tube."
        }
    },
    "kNN": {
        "name": "K-Nearest Neighbours Regressor",
        "description": "Predicts by averaging the target values of the k closest training samples.",
        "history": "Proposed by Fix and Hodges in 1951.",
        "equation": "ŷ = (1/k) Σᵢ yᵢ",
        "parameters": {
            "n_neighbors": "Number of neighbours k.",
            "weights": "Weight function (uniform or distance).",
            "p": "Power parameter for Minkowski distance."
        }
    },
    "DT": {
        "name": "Decision Tree Regressor",
        "description": "Partitions the feature space with binary splitting rules to form a piecewise-constant predictor.",
        "history": "Introduced as CART by Breiman et al. in 1984.",
        "equation": "—",
        "parameters": {
            "max_depth": "Maximum tree depth.",
            "min_samples_split": "Minimum samples required to split an internal node.",
            "min_samples_leaf": "Minimum samples required at a leaf node.",
            "max_features": "Number of features considered at each split."
        }
    },
    "AdaBoost": {
        "name": "AdaBoost Regressor",
        "description": "Combines weak learners sequentially with adaptive sample weighting.",
        "history": "Developed by Freund and Schapire in 1997.",
        "equation": "F(x) = Σₘ αₘ hₘ(x)",
        "parameters": {
            "n_estimators": "Number of weak learners.",
            "learning_rate": "Weight shrinkage applied to each learner.",
            "loss": "Loss function (linear, square, or exponential)."
        }
    },
    "RF": {
        "name": "Random Forest Regressor",
        "description": "Averages predictions of multiple decision trees trained on bootstrap samples with random feature subsets.",
        "history": "Introduced by Leo Breiman in 2001.",
        "equation": "ŷ = (1/M) Σₘ Tₘ(x)",
        "parameters": {
            "n_estimators": "Number of trees.",
            "max_depth": "Maximum tree depth.",
            "min_samples_split": "Minimum samples to split a node.",
            "min_samples_leaf": "Minimum samples at a leaf.",
            "max_features": "Feature subset size per split."
        }
    },
    "ET": {
        "name": "Extra Trees Regressor",
        "description": "Builds trees with fully randomised splitting thresholds and averages their predictions.",
        "history": "Proposed by Geurts et al. in 2006.",
        "equation": "ŷ = (1/M) Σₘ Tₘ(x)",
        "parameters": {
            "n_estimators": "Number of trees.",
            "max_depth": "Maximum tree depth.",
            "min_samples_split": "Minimum samples to split a node.",
            "min_samples_leaf": "Minimum samples at a leaf.",
            "max_features": "Feature subset size per split."
        }
    },
    "GBDT": {
        "name": "Gradient Boosting Regressor",
        "description": "Friedman's original gradient boosting framework with decision tree weak learners.",
        "history": "Published by Jerome H. Friedman in 2001.",
        "equation": "Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)",
        "parameters": {
            "n_estimators": "Number of boosting stages.",
            "learning_rate": "Step size per stage (η).",
            "max_depth": "Maximum tree depth.",
            "subsample": "Fraction of samples used per stage."
        }
    },
    "HGB": {
        "name": "Histogram Gradient Boosting Regressor",
        "description": "Histogram-based GBDT variant with native support for missing values.",
        "history": "Introduced in scikit-learn 0.21 (2018).",
        "equation": "Leaf-wise GBDT update: Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)",
        "parameters": {
            "learning_rate": "Step size per stage.",
            "max_iter": "Maximum number of boosting iterations.",
            "max_leaf_nodes": "Maximum number of leaves per tree.",
            "l2_regularization": "L2 regularisation strength."
        }
    },
    "KR": {
        "name": "Kernel Ridge Regression",
        "description": "Extends Ridge regression to non-linear feature spaces via the kernel trick.",
        "history": "Proposed by Saunders et al. in 1998.",
        "equation": "minimise ||y – Kα||² + α αᵀ K",
        "parameters": {
            "alpha": "L2 regularisation strength.",
            "kernel": "Kernel type (linear, rbf, polynomial).",
            "gamma": "Scale parameter for the RBF kernel."
        }
    },
    "PolyR": {
        "name": "Polynomial Ridge Regression",
        "description": "Pipeline that expands features with polynomial terms and applies Ridge regularisation.",
        "history": "Polynomial expansion is a classical technique; Ridge regularisation was formalised in the 1970s.",
        "equation": "y = β₀ + Σᵢ βᵢ xᵢ + Σ_{i<j} β_{ij} xᵢ xⱼ + …",
        "parameters": {
            "poly__degree": "Highest polynomial degree.",
            "poly__interaction_only": "Include only interaction terms if True.",
            "ridge__alpha": "Ridge penalty coefficient (α)."
        }
    },
    "ExtraTrees": {
        "name": "Extra Trees Regressor",
        "description": "Builds an ensemble of fully randomised trees and averages their outputs.",
        "history": "Proposed by Geurts, Ernst and Wehenkel in 2006.",
        "equation": "ŷ = (1/M) Σₘ Tₘ(x)",
        "parameters": {
            "n_estimators": "Number of trees.",
            "max_depth": "Maximum tree depth.",
            "min_samples_split": "Minimum samples to split a node.",
            "min_samples_leaf": "Minimum samples at a leaf.",
            "max_features": "Feature subset size per split."
        }
    },
    "GPR": {
        "name": "Gaussian Process Regressor",
        "description": "Non-parametric Bayesian regression that places a prior over functions.",
        "history": "Based on Rasmussen and Williams (2006).",
        "equation": "f(x) ~ GP(m(x), k(x, x'))",
        "parameters": {
            "kernel": "Covariance function (RBF, Matérn, etc.).",
            "alpha": "Noise variance added to the kernel diagonal."
        }
    },
    "Stack": {
        "name": "Stacking Regressor",
        "description": "Combines predictions of multiple base models using a meta-learner.",
        "history": "Proposed by David H. Wolpert in 1992.",
        "equation": "F(x) = g(h₁(x), h₂(x), …, hₙ(x))",
        "parameters": {
            "final_estimator__alpha": "Ridge penalty for the meta-learner.",
            "passthrough": "Append original features to meta-learner input if True."
        }
    },
    "Quantile": {
        "name": "Quantile Regressor",
        "description": "Minimises the pinball loss to estimate a specified conditional quantile.",
        "history": "Introduced by Roger Koenker and Gilbert Bassett in 1978.",
        "equation": "minimise Σ ρ_τ(yᵢ – xᵢᵀβ)",
        "parameters": {
            "quantile": "Target quantile τ (0–1).",
            "alpha": "L2 regularisation strength."
        }
    },
    "Poisson": {
        "name": "Poisson Regressor",
        "description": "GLM with a log link function and Poisson distribution, suited for count-like targets.",
        "history": "Builds on Sir Ronald A. Fisher's GLM framework (1922).",
        "equation": "log(μ) = Xβ",
        "parameters": {
            "alpha": "L2 regularisation strength.",
            "max_iter": "Maximum number of solver iterations."
        }
    },
    "Tweedie": {
        "name": "Tweedie Regressor",
        "description": "GLM based on the Tweedie distribution family, covering a range of variance structures.",
        "history": "Builds on Dunn and Smyth (2005).",
        "equation": "V(y) = φ μ^p; link can be log or identity.",
        "parameters": {
            "power": "Distribution power parameter p.",
            "alpha": "L2 regularisation strength."
        }
    },
    "Gamma": {
        "name": "Gamma Regressor",
        "description": "GLM with a Gamma distribution and log link, suitable for positive continuous targets.",
        "history": "A classical application within the GLM methodology.",
        "equation": "g(μ) = Xβ, g = log link",
        "parameters": {
            "alpha": "L2 regularisation strength.",
            "max_iter": "Maximum number of solver iterations."
        }
    },
    "OMP": {
        "name": "Orthogonal Matching Pursuit",
        "description": "Greedily selects features that best explain residuals at each step.",
        "history": "Rooted in Mallat and Zhang's sparse coding work (1993).",
        "equation": "—",
        "parameters": {
            "n_nonzero_coefs": "Maximum number of non-zero coefficients."
        }
    },
    "PA": {
        "name": "Passive Aggressive Regressor",
        "description": "Online learning algorithm that updates aggressively on each misclassified sample.",
        "history": "Proposed by Crammer et al. in 2006.",
        "equation": "β_{t+1} = argmin_β ½||β – β_t||² + C L(β; x_t, y_t)",
        "parameters": {
            "C": "Regularisation parameter for the hinge loss.",
            "epsilon": "Insensitivity threshold for regression."
        }
    },
    "CAT": {
        "name": "CatBoost Regressor",
        "description": "Gradient boosting with ordered boosting and native categorical feature support.",
        "history": "Developed by Yandex in 2017.",
        "equation": "Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x), with ordered boosting",
        "parameters": {
            "iterations": "Total number of boosting iterations.",
            "learning_rate": "Step size per iteration (η).",
            "depth": "Tree depth.",
            "l2_leaf_reg": "L2 regularisation strength.",
            "bagging_temperature": "Bayesian bootstrap temperature."
        }
    },
    "NGBoost": {
        "name": "Natural Gradient Boosting",
        "description": "Probabilistic gradient boosting that models the full predictive distribution using natural gradient descent.",
        "history": "Introduced by Duan et al. (2020) at Stanford.",
        "equation": "P(y|x) = Dist(θ(x)), where θ is learned via natural gradient boosting",
        "parameters": {
            "n_estimators": "Number of boosting rounds.",
            "learning_rate": "Shrinkage factor per round.",
            "minibatch_frac": "Fraction of data used per iteration.",
            "col_sample": "Fraction of features sampled per tree."
        }
    },
    "XGBoost_RF": {
        "name": "XGBoost Random Forest",
        "description": "XGBoost operating in Random Forest mode: each tree is built independently with subsampling, combining boosting infrastructure with bagging.",
        "history": "XGBoost RF mode introduced by Chen & Guestrin (2016); RF mode formalised in XGBoost 1.0.",
        "equation": "F(x) = (1/T) Σ hₜ(x), each tree trained independently with row/col subsampling",
        "parameters": {
            "n_estimators": "Number of trees.",
            "max_depth": "Maximum tree depth.",
            "subsample": "Row subsampling ratio.",
            "colsample_bytree": "Column subsampling ratio per tree.",
            "num_parallel_tree": "Number of parallel trees per round (RF depth)."
        }
    },
    "Voting": {
        "name": "Voting Regressor",
        "description": "Ensemble that averages predictions from multiple diverse base models (XGBoost, LightGBM, CatBoost, Random Forest).",
        "history": "Classic ensemble averaging method; sklearn implementation by Pedregosa et al. (2011).",
        "equation": "ŷ = (1/N) Σᵢ fᵢ(x), where fᵢ are base estimators",
        "parameters": {
            "xgb__n_estimators": "XGBoost number of trees.",
            "xgb__learning_rate": "XGBoost learning rate.",
            "lgb__n_estimators": "LightGBM number of trees.",
            "rf__n_estimators": "Random Forest number of trees."
        }
    },

}