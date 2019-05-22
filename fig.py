import click
import numpy as np
import matplotlib.pyplot as plt

from data import load_data


@click.command()
@click.option("--save/--show",
              help='Save the image to file.')
@click.option("--feature",
              default=0,
              help='Index of feature to use in plot.')
@click.argument("data_name")
@click.argument("y_name")
def main(save, feature, data_name, y_name):

    x_index = feature

    x, y = load_data("data/"+data_name)
    yp = np.load(y_name)

    fig, ax = plt.subplots()

    print(x.shape, y.shape)
    ax.scatter(x[:, x_index], y, color="b")

    print(x.shape, yp.shape)
    ax.scatter(x[:, x_index], yp, color="r")

    if save:
        plt.savefig("{}_{}_{}.eps".format(data_name, x_index, y_name),
                    bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    main()
