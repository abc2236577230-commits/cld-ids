"""Model factories for the unified experiment matrix."""
import numpy as np
from river import drift, forest, tree
from sklearn.neural_network import MLPClassifier


def make_arf(n_models=10, seed=42):
    """Off-the-shelf Adaptive Random Forest (river). Defaults kept except
    leaf_prediction='mc' to match our HoeffdingTree configuration."""
    return forest.ARFClassifier(n_models=n_models, seed=seed,
                                leaf_prediction="mc")


def make_mlp(lr=5e-4, hidden=(32, 16), batch=64, seed=42):
    return MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                         learning_rate_init=lr, learning_rate="constant",
                         max_iter=1, random_state=seed, batch_size=batch)


def make_ht(grace=50, delta=1e-5):
    return tree.HoeffdingTreeClassifier(grace_period=grace, delta=delta,
                                        leaf_prediction="mc")


def make_periodic(period=5000, window=2000, lr=1e-3, seed=42):
    """Periodic retraining baseline: every `period` samples, refit an MLP on
    the most recent `window` samples."""
    return {"period": period, "window": window, "lr": lr, "seed": seed}


def make_drc_ht(delta=0.002):
    """river's off-the-shelf drift-retraining classifier around HoeffdingTree
    (background training off -> reset-on-drift, i.e. no warm-start)."""
    return drift.DriftRetrainingClassifier(
        model=tree.HoeffdingTreeClassifier(grace_period=50, delta=1e-5,
                                           leaf_prediction="mc"),
        drift_detector=drift.ADWIN(delta=delta),
        train_in_background=False,
    )


def make_static_mlp(lr=1e-3, seed=42):
    return MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                         learning_rate_init=lr, max_iter=30,
                         random_state=seed, batch_size=256)


class PeriodicMLP:
    """Online MLP with periodic refitting on a sliding labeled buffer."""

    def __init__(self, period=5000, window=2000, lr=1e-3, seed=42, batch=256,
                 n_features=None):
        self.period = period
        self.window = window
        self.lr = lr
        self.seed = seed
        self.batch = batch
        self.buf_x, self.buf_y = [], []
        self.model = None
        self.fitted = False
        self.n = 0
        self.n_features = n_features

    def predict(self, xn):
        if not self.fitted:
            return 0
        return int(self.model.predict(xn)[0])

    def update(self, xn, y):
        self.n += 1
        self.buf_x.append(xn[0])
        self.buf_y.append(int(y))
        if len(self.buf_x) > self.window:
            self.buf_x = self.buf_x[-self.window:]
            self.buf_y = self.buf_y[-self.window:]
        if self.n % self.period == 0 and len(self.buf_x) >= 64:
            Xb = np.array(self.buf_x, dtype=np.float32)
            yb = np.array(self.buf_y, dtype=np.int8)
            self.model = MLPClassifier(
                hidden_layer_sizes=(32, 16), activation="relu",
                learning_rate_init=self.lr, max_iter=30,
                random_state=self.seed, batch_size=self.batch)
            self.model.fit(Xb, yb)
            self.fitted = True


class DriftAware:
    """ADWIN error channel + reset + warm-start replay (+ cooldown, + warm-up)."""

    def __init__(self, make_base, delta=0.002, cooldown=5000, warmup=512, batch=64,
                 start_after=0, finetune_epochs=0, finetune_lr=1e-2):
        self.make_base = make_base
        self.delta = delta
        self.cooldown = cooldown
        self.warmup = warmup
        self.batch = batch
        self.start_after = start_after
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.model = make_base()
        self.fitted = False
        self.adwin = drift.ADWIN(delta=delta)
        self.warm = []
        self.last_trigger = -cooldown

    def predict(self, x):
        """x is xn (numpy row) for sklearn models, xd (dict) for river models."""
        if not self.fitted:
            return 0
        if hasattr(self.model, "predict_one"):
            return self.model.predict_one(x) or 0
        return int(self.model.predict(x)[0])

    def _replay(self):
        if hasattr(self.model, "learn_one"):
            for x, y in self.warm:
                self.model.learn_one(x, y)
            self.fitted = True
            return
        for s in range(0, len(self.warm) - self.batch + 1, self.batch):
            seg = self.warm[s : s + self.batch]
            Xb = np.array([np.asarray(v[0]).reshape(-1) for v in seg], dtype=np.float32)
            self.model.partial_fit(Xb, np.array([v[1] for v in seg]), classes=[0, 1])
        if self.finetune_epochs > 0 and len(self.warm) >= 2 * self.batch:
            from sklearn.neural_network import MLPClassifier
            Xb = np.array([np.asarray(v[0]).reshape(-1) for v in self.warm], dtype=np.float32)
            yb = np.array([v[1] for v in self.warm])
            m2 = MLPClassifier(hidden_layer_sizes=self.model.hidden_layer_sizes,
                               activation="relu", learning_rate_init=self.finetune_lr,
                               learning_rate="constant", max_iter=self.finetune_epochs,
                               random_state=0, batch_size=self.batch)
            m2.partial_fit(Xb, yb, classes=[0, 1])
            self.model = m2
        self.fitted = True

    def update_detector(self, error):
        if self.fitted:
            self.adwin.update(int(error))

    def maybe_reset(self, i):
        """Returns True (and resets) when ADWIN fires outside warm-up/cooldown."""
        if not self.fitted or not self.adwin.drift_detected:
            return False
        if i + 1 <= self.start_after:
            return False
        if i + 1 - self.last_trigger <= self.cooldown:
            return False
        self.last_trigger = i + 1
        self.model = self.make_base()
        self.fitted = False
        self._replay()
        self.warm = []
        self.adwin = drift.ADWIN(delta=self.delta)
        return True

    def push_warm(self, x, y):
        self.warm.append((x, int(y)))
        if len(self.warm) > self.warmup:
            self.warm.pop(0)