from importlib import import_module

import click
import numpy as np
from keras import losses


from data import load_data
from loss import CustomLoss
from utils import mean_squared_error, trimmed_mean_squared_error

# seed(42)
# set_random_seed(42)


def fit_n(n, model_generator, loss, x, y):
    """ Fits n models and returns the one with the lowest cost. """

    n_inputs = x.shape[1]
    models = [model_generator(loss, n_inputs) for _ in range(n)]
    losses = []
    for model in models:
        history = model.fit(x, y,
                            batch_size=32,
                            epochs=20000,
                            verbose=1)
        loss_i = history.history["loss"][-1]
        losses.append(loss_i)

    print(losses)
    winner = models[np.argmin(losses)]
    return winner


@click.group()
def test():
    pass


@test.command()
@click.option("--label", default="test",
              help="Label to prepend to output file names.")
@click.argument("data_name")
@click.argument("tau1", type=float)
@click.argument("tau2", type=float)
@click.argument("model_name")
def quantiles(data_name, label, tau1, tau2, model_name):

    x, y = load_data("data/"+data_name)

    mod = import_module(model_name)
    model = mod.model

    # model 1 - upper bound
    loss = CustomLoss(tau1)
    model1 = fit_n(5, model, loss.loss, x, y)
    ym1 = model1.predict(x)
    model1.save("mlp_{}_{}_{}.h5".format(data_name, label, tau1))
    np.save("mlp_{}_{}_y_{}".format(data_name, label, tau1), ym1)

    # model 2 - lower bound
    loss = CustomLoss(tau2)
    model2 = fit_n(5, model, loss.loss, x, y)
    ym2 = model2.predict(x)
    model2.save("mlp_{}_{}_{}.h5".format(data_name, label, tau2))
    np.save("mlp_{}_{}_y_{}".format(data_name, label, tau2), ym2)


@test.command()
def crossvalidation():
    click.echo("hello crossvalidation")


@test.command()
@click.option("--label", default="test",
              help="Label to prepend to output file names.")
@click.argument("data_name")
@click.argument("model_name")
def simple_model(label, data_name, model_name):

    x, y = load_data("data/"+data_name)

    mod = import_module(model_name)
    model = mod.model

    # model
    trained_model = fit_n(5, model, losses.mean_squared_error, x, y)
    ym1 = trained_model.predict(x)

    trained_model.save("mlp_{data}_{label}_plain.h5".format(
        data=data_name, label=label))
    np.save("mlp_{data}_{label}_y_plain".format(
        data=data_name, label=label), ym1)

    # calculate final loss
    ym1 = ym1.squeeze()
    print("MSE:  ", mean_squared_error(y, ym1))
    print("TMSE: ", trimmed_mean_squared_error(y, ym1))


@test.command()
@click.option("--label", default="test",
              help="Label to prepend to output file names.")
@click.argument("data_name")
@click.argument("lower_name")
@click.argument("upper_name")
@click.argument("model_name")
def trimmed_model(data_name, label, lower_name, upper_name, model_name):

    lower = np.load(lower_name)
    upper = np.load(upper_name)

    x1, y1 = load_data("data/"+data_name)
    x, y = load_data("data/"+data_name, lower, upper)

    mod = import_module(model_name)
    model = mod.model

    # model
    trained_model = fit_n(5, model, losses.mean_squared_error, x, y)
    ym1 = trained_model.predict(x1)

    trained_model.save(
        "mlp_{data}_{label}_trimmed_{lower}_{upper}.h5".format(
            data=data_name,
            label=label,
            lower=lower_name,
            upper=upper_name))
    np.save("mlp_{data}_{label}_y_trimmed_{lower}_{upper}".format(
        data=data_name, label=label, lower=lower_name, upper=upper_name),
        ym1)

    # calculate final loss
    ym1 = ym1.squeeze()
    click.echo("MSE:  {}".format(mean_squared_error(y1, ym1)))
    click.echo("TMSE: {}".format(trimmed_mean_squared_error(y1, ym1)))


if __name__ == "__main__":
    test()
