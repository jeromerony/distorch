from matplotlib import pyplot as plt


def single_pixel_representation():
    fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained', figsize=(4, 2))
    ax1: plt.Axes
    ax2: plt.Axes

    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 2)
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_yticks([0, 1, 2, 3])
    ax1.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    ax1.grid(alpha=0.3)
    points = ((1, 1), (2, 1), (2, 2), (1, 2))
    surface = plt.Polygon(points, linewidth=2, hatch='xx', color='C0', alpha=0.5, fill=None, edgecolor=None, zorder=5)
    contour = plt.Polygon(points, linewidth=2, color='C0', fill=None, zorder=10)
    ax1.scatter(*zip(*points), c='C0')
    ax1.add_artist(surface)
    ax1.add_artist(contour)
    ax1.set_aspect('equal')

    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 2)
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_yticks([0, 1, 2, 3])
    ax2.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    ax2.grid(alpha=0.3)
    ax2.scatter(1.5, 1.5, c='C1')
    surface = plt.Polygon(points, linewidth=0, hatch='xx', color='C1', alpha=0.5, fill=None)
    ax2.set_aspect('equal')
    ax2.add_artist(surface)

    return fig


def vertices_approx_ASD():
    fig, ax = plt.subplots(layout='constrained', figsize=(3.2, 2.4))
    ax: plt.Axes

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3])
    ax.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    ax.grid(alpha=0.3)

    points_b = ((1, 1), (2, 1), (2, 2), (1, 2))
    contour_b = plt.Polygon(points_b, linewidth=2, linestyle='--', color='C1', fill=None, zorder=10)
    ax.scatter(*zip(*points_b), c='C1', label='$a$', zorder=10)
    ax.set_aspect('equal')
    ax.add_artist(contour_b)

    points_a = ((1, 1), (3, 1), (3, 2), (1, 2))
    contour_a = plt.Polygon(points_a, linewidth=2, color='C0', fill=None, zorder=5)
    ax.scatter(*zip(*points_a), c='C0', label='$b$', zorder=5)
    ax.add_artist(contour_a)

    ax.legend()

    ax.text(2.05, 1.45, '$i_1$', c='C1')
    ax.text(2.45, 2.1, '$i_2$', c='C0')
    ax.text(3.05, 1.45, '$i_3$', c='C0')
    ax.text(2.45, 0.8, '$i_4$', c='C0')

    return fig


def diagonal_pixels():
    fig, ax = plt.subplots(layout='constrained', figsize=(3.2, 3.2))
    ax: plt.Axes

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3])
    ax.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    points = ((1, 1), (2, 1), (2, 2), (1, 2), (2, 2), (3, 2), (3, 3), (2, 3))
    ax.add_artist(plt.Polygon(points[:4], linewidth=2, color='C0', fill=None, hatch='x'))
    ax.add_artist(plt.Polygon(points[4:], linewidth=2, color='C0', fill=None, hatch='x'))
    ax.scatter(*zip(*points), c='C0')

    return fig

def anisotropic_boundary():
    fig, ax = plt.subplots(layout='constrained', figsize=(3.2, 1.2))
    ax: plt.Axes

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3])
    ax.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    ax.grid(alpha=0.3)

    points = ((1, 1), (2, 1), (3, 1), (3, 2), (2, 2), (1, 2))
    ax.add_artist(plt.Polygon(points, linewidth=2, color='C0', fill=None, hatch='x'))
    ax.scatter(*zip(*points), c='C0')

    return fig


if __name__ == '__main__':
    fig = single_pixel_representation()
    fig.savefig('single_pixel.png')

    fig = vertices_approx_ASD()
    fig.savefig('vertices_approx_ASD.png')

    fig = diagonal_pixels()
    fig.savefig('diagonal_pixels.png')

    fig = anisotropic_boundary()
    fig.savefig('anisotropic_boundary.png')