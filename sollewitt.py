import numpy as np
import itertools
import plotly.graph_objects as go

# --- CUBE GEOMETRY DEFINITION ---
# A cube has 8 vertices. We define their (x, y, z) coordinates.
vertices = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
)
# A cube has 12 edges, indexed 0-11. Each edge connects two vertices.
edges = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
)


def plot_incomplete_cube(visibility_vector):
    """
    Takes a binary vector and draws the corresponding cube edges in an interactive 3D plot.

    Args:
        visibility_vector (list or np.array): An array of 12 numbers (0s or 1s).
    """
    if len(visibility_vector) != 12:
        raise ValueError("visibility_vector must have a length of 12.")

    # These lists will hold the coordinates for the lines to be drawn.
    # We use 'None' to create breaks in the line, so each edge is separate.
    x_coords = []
    y_coords = []
    z_coords = []

    # Iterate through each of the 12 edges
    for i, is_visible in enumerate(visibility_vector):
        # If the vector at this index is 1, we draw the edge
        if is_visible == 1:
            # Get the indices of the start and end vertices for the current edge
            vertex_indices = edges[i]
            start_vertex = vertices[vertex_indices[0]]
            end_vertex = vertices[vertex_indices[1]]

            # Add start vertex coordinates
            x_coords.append(start_vertex[0])
            y_coords.append(start_vertex[1])
            z_coords.append(start_vertex[2])

            # Add end vertex coordinates
            x_coords.append(end_vertex[0])
            y_coords.append(end_vertex[1])
            z_coords.append(end_vertex[2])

            # Add None to break the line before the next edge
            x_coords.append(None)
            y_coords.append(None)
            z_coords.append(None)

    # Create the figure
    fig = go.Figure()

    # Add the 3D line trace for the cube edges
    fig.add_trace(
        go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="lines",
            line=dict(color="#22d3ee", width=8),
            hoverinfo="none",
        )
    )

    # Update the layout for a clean, cube-like appearance
    fig.update_layout(
        title="Interactive 3D Cube",
        paper_bgcolor="#111827",  # Dark background
        plot_bgcolor="#111827",
        scene=dict(
            xaxis=dict(visible=False, range=[-0.2, 1.2]),
            yaxis=dict(visible=False, range=[-0.2, 1.2]),
            zaxis=dict(visible=False, range=[-0.2, 1.2]),
            # This ensures the cube looks like a cube and not a stretched box
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # Show the interactive plot
    fig.show()


if __name__ == "__main__":
    example_binary_vector = [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1]
    print(f"Visualizing cube for vector: {example_binary_vector}")
    plot_incomplete_cube(example_binary_vector)
#
# sides = 12
# all_combinations = np.array(list(itertools.product([0, 1], repeat=sides)))
# minimal_moves = np.array(
#     [
#         [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8],
#         [5, 9, 6, 1, 0, 8, 10, 2, 4, 11, 7, 3],
#         [2, 6, 10, 7, 3, 1, 9, 11, 0, 5, 8, 4],
#     ],
#     dtype=np.int8,
# )
# actions = (
#     (),
#     (0,),
#     (0, 0),
#     (0, 0, 0),
#     (1,),
#     (1, 1),
#     (1, 1, 1),
#     (2,),
#     (2, 2),
#     (2, 2, 2),
#     (0, 2),
#     (0, 2, 2),
#     (0, 2, 2, 2),
#     (0, 0, 2),
#     (0, 0, 2, 2, 2),
#     (0, 0, 0, 2),
#     (0, 0, 0, 2, 2),
#     (0, 0, 0, 2, 2, 2),
#     (1, 2),
#     (1, 2, 2),
#     (1, 2, 2, 2),
#     (1, 1, 1, 2),
#     (1, 1, 1, 2, 2),
#     (1, 1, 1, 2, 2, 2),
# )
#
# initial_state = np.arange(sides, dtype=np.int8)
# all_moves_list = []
# for action_sequence in actions:
#     current_state = initial_state.copy()
#     if not action_sequence == ():
#         for move_id in action_sequence:
#             current_state = current_state[minimal_moves[move_id]]
#     all_moves_list.append(current_state)
# all_moves = np.stack(all_moves_list)
# print(f"Generated a {all_moves.shape} NumPy array with all move mappings.\n")
# print(all_moves)
# # print(all_combinations)
