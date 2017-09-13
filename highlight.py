import matplotlib.pyplot as plt

def label_area(points, highlighted_word):
    target_x = 0
    target_y = 0

    # Here, we're just finding the coordinates of the target word
    for _i, point in points.iterrows():
        if point.word == highlighted_word:
            target_x = point.x
            target_y = point.y

    print("Finding word...")
    # Just for better colors
    plot_region(
        x_bounds=(target_x - 0.5, target_x + 0.5), 
        y_bounds=(target_y - 0.5, target_y + 0.5), 
        points=points,
        highlighted_word=highlighted_word,
    )

def plot_region(x_bounds, y_bounds, points, highlighted_word):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))

    for _i, point in slice.iterrows():
        if (point.word == highlighted_word):
            ax.text(point.x + 0.005, point.y + 0.005, point.word, color='red', fontsize=11)
        else:
            ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)