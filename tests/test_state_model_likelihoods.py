import numpy as np
from bayesgbm import GaussianPrior, Priors, StateModel


def test_bernoulli_likelihood_matches_formula():
    model = StateModel(
        evolution=lambda x, th, u, y: x,
        observation=lambda x, ph, u: 0.8,
        family="bernoulli",
        priors=Priors(GaussianPrior([0], [0]), GaussianPrior([0], [0])),
        initial_state=[0],
    )
    out = model.evaluate([0, 0], {"y": np.array([1, 0]), "u": None})
    np.testing.assert_allclose(out["loglik"], [np.log(.8), np.log(.2)])


def test_categorical_likelihood_matches_formula():
    model = StateModel(
        evolution=lambda x, th, u, y: x,
        observation=lambda x, ph, u: np.array([.2, .3, .5]),
        family="categorical",
        priors=Priors(GaussianPrior([0], [0]), GaussianPrior([0], [0])),
        initial_state=[0],
    )
    out = model.evaluate([0, 0], {"y": np.array([2, 1]), "u": None})
    np.testing.assert_allclose(out["loglik"], [np.log(.5), np.log(.3)])


def test_gaussian_likelihood_matches_formula():
    model = StateModel(
        evolution=lambda x, th, u, y: x,
        observation=lambda x, ph, u: np.array([ph[0]]),
        family="gaussian",
        priors=Priors(GaussianPrior([0], [0]), GaussianPrior([0], [1])),
        initial_state=[0],
        observation_covariance=4.0,
    )
    y = np.array([2.0])
    out = model.evaluate([0, 0], {"y": y, "u": None})
    expected = -0.5 * (np.log(2*np.pi*4) + 1.0)
    np.testing.assert_allclose(out["loglik"][0], expected)


def test_generic_u_dictionary_is_sliced_without_reserved_names():
    seen = []
    def evo(x, th, u, y):
        seen.append((u["abc"], u["condition"]))
        return x
    model = StateModel(
        evo, lambda x, ph, u: .5, "bernoulli",
        Priors(GaussianPrior([0],[0]), GaussianPrior([0],[0])), [0]
    )
    model.evaluate([0,0], {"y": np.array([0,1]), "u": {"abc": np.array([3,4]), "condition": "A"}})
    assert seen == [(3, "A"), (4, "A")]
