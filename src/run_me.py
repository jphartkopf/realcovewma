import numpy as np
from utils import duplication_matrix
from losses import frob, stein, vnd
from models import EWMA, UhligExtension


if __name__ == "__main__":
    # load data
    raw_cv_data = np.genfromtxt("../data/CV.csv", delimiter=",")
    k, t = raw_cv_data.shape
    n = int(-0.5 + np.sqrt(0.25 + 2*k))
    dup = duplication_matrix(n)
    y = np.array(dup @ raw_cv_data).reshape((n, n, t))

    yhat = np.zeros(shape=y.shape)
    yhat[:, :, 0] = np.eye(n)

    # Choose your model
    # Note: UE is fitted inside for-loop, EWMA only once
    # mdl = EWMA()
    # mdl.fit(lam=0.96)
    mdl = UhligExtension()

    for s in [1/3, 1/2, 2/3]:
        tt = int(np.floor(t*s))

        if mdl.__class__.__name__ == "UhligExtension":
            mdl.fit(y[:, :, :tt], 12, 15)

        # predictions
        for t_ in np.arange(1, t):
            yhat[:, :, t_] = mdl.predict(y[:, :, t_ - 1], yhat[:, :, t_ - 1])

        # loss calculations
        frob_model = 0.
        frob_benchmark = 0.
        st_model = 0.
        st_benchmark = 0.
        vnd_model = 0.
        vnd_benchmark = 0.
        for t_ in np.arange(tt, t):
            frob_benchmark += frob(y[:, :, t_], y[:, :, t_-1])
            frob_model += frob(y[:, :, t_], yhat[:, :, t_])
            st_benchmark += stein(y[:, :, t_], y[:, :, t_-1])
            st_model += stein(y[:, :, t_], yhat[:, :, t_])
            vnd_benchmark += vnd(y[:, :, t_], y[:, :, t_-1])
            vnd_model += vnd(y[:, :, t_], yhat[:, :, t_])

        print("")
        print(f"Out-of-sample evaluation for T_0 = {s:.2f}*T")
        print(f"Model: {mdl.__class__.__name__} (lam={mdl.lam:.4f})")
        print(f"Frob: {frob_model / frob_benchmark:.3f}")
        print(f"Stein: {st_model / st_benchmark:.3f}")
        print(f"VND: {vnd_model / vnd_benchmark:.3f}")
        print(f"{'-'*30}")
