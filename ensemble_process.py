import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from ensembles import PlaceCellEnsemble, HeadDirectionCellEnsemble

# Include the classes provided in the user's prompt here:
# one_hot_max, softmax, softmax_sample, CellEnsemble, 
# PlaceCellEnsemble, HeadDirectionCellEnsemble

# --- 1. Setup and Data Generation ---

# Parameters
n_place_cells = 256
n_hd_cells = 128
pos_min = -5
pos_max = 5
stdev = 0.35
concentration = 20
seed = 42
batch_size = 1
n_steps = 100

# Create ensembles
place_ensemble = PlaceCellEnsemble(n_place_cells, stdev=stdev, pos_min=pos_min, pos_max=pos_max, seed=seed, soft_targets="softmax")
hd_ensemble = HeadDirectionCellEnsemble(n_hd_cells, concentration=concentration, seed=seed, soft_targets="softmax")

# Generate synthetic trajectory (a simple circle)
t = np.linspace(0, 2*np.pi, n_steps)
radius = 3
x = radius * np.cos(t)
y = radius * np.sin(t)
positions = np.stack([x, y], axis=1)  # Shape: (n_steps, 2)
angles = np.arctan2(y, x)  # Shape: (n_steps,)

# Reshape for TensorFlow (Batch, Time, ...)
positions_tf = tf.constant(positions[np.newaxis, ...], dtype=tf.float32)
angles_tf = tf.constant(angles[np.newaxis, :, np.newaxis], dtype=tf.float32)


# --- 2. Calculate Cell Activities and Targets ---

# Place Cells
place_targets_tf = place_ensemble.get_targets(positions_tf)
place_log_pdf_tf = place_ensemble.unnor_logpdf(positions_tf)

# Head Direction Cells
hd_targets_tf = hd_ensemble.get_targets(angles_tf)
hd_log_pdf_tf = hd_ensemble.unnor_logpdf(angles_tf)

# Execute TF graph
with tf.Session() as sess:
    place_targets, place_log_pdf = sess.run([place_targets_tf, place_log_pdf_tf])
    hd_targets, hd_log_pdf = sess.run([hd_targets_tf, hd_log_pdf_tf])

# Remove batch dimension for visualization
place_targets = place_targets[0]  # Shape: (n_steps, n_place_cells)
place_log_pdf = place_log_pdf[0]  # Shape: (n_steps, n_place_cells)
hd_targets = hd_targets[0]        # Shape: (n_steps, n_hd_cells)
hd_log_pdf = hd_log_pdf[0]        # Shape: (n_steps, n_hd_cells)


# --- 3. Visualization ---

fig = plt.figure(figsize=(16, 10))

# --- a) Place Cell Firing Fields ---
ax1 = fig.add_subplot(2, 3, 1)
ax1.set_title("Place Cell Centers and Trajectory")
ax1.set_xlim(pos_min, pos_max)
ax1.set_ylim(pos_min, pos_max)
ax1.set_aspect('equal')

# Plot all place cell centers
ax1.scatter(place_ensemble.means[:, 0], place_ensemble.means[:, 1], s=10, c='gray', alpha=0.5, label="Place Cell Centers")

# Highlight a few example place cells and their firing fields
example_cells = [0, n_place_cells // 2, n_place_cells - 1]
colors = ['r', 'g', 'b']
for i, cell_idx in enumerate(example_cells):
    mean = place_ensemble.means[cell_idx]
    # Firing field is a Gaussian; plot a circle at 1 standard deviation
    circle = Circle(mean, stdev, color=colors[i], alpha=0.3)
    ax1.add_patch(circle)
    ax1.scatter(mean[0], mean[1], s=50, c=colors[i], label=f"Cell {cell_idx}")

# Plot the trajectory
ax1.plot(positions[:, 0], positions[:, 1], 'k-', linewidth=2, label="Trajectory")
ax1.legend()

# --- b) Activity of Example Place Cells over Time ---
ax2 = fig.add_subplot(2, 3, 2)
ax2.set_title("Activity of Example Place Cells")
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Unnormalized Log Probability")
for i, cell_idx in enumerate(example_cells):
    # Activity is the unnormalized log PDF
    activity = place_log_pdf[:, cell_idx]
    ax2.plot(activity, color=colors[i], label=f"Cell {cell_idx}")
ax2.legend()

# --- c) Decoded Position (Target Distribution) at a Specific Time ---
time_idx = n_steps // 4  # Visualize at 1/4 of the trajectory
ax3 = fig.add_subplot(2, 3, 3)
ax3.set_title(f"Decoded Position at t={time_idx}")
ax3.set_xlim(pos_min, pos_max)
ax3.set_ylim(pos_min, pos_max)
ax3.set_aspect('equal')

# The target distribution is a softmax over the place cells.
# We can visualize this by coloring the place cell centers by their target probability.
target_probs = place_targets[time_idx]
sc = ax3.scatter(place_ensemble.means[:, 0], place_ensemble.means[:, 1], c=target_probs, cmap='viridis', s=30)
plt.colorbar(sc, ax=ax3, label="Probability")

# Mark true position
true_pos = positions[time_idx]
ax3.plot(true_pos[0], true_pos[1], 'r*', markersize=15, label="True Position")
ax3.legend()


# --- d) Head Direction Cell Tuning Curves ---
ax4 = fig.add_subplot(2, 3, 4)
ax4.set_title("Head Direction Cell Tuning Curves")
ax4.set_xlabel("Angle (radians)")
ax4.set_ylabel("Unnormalized Log Probability")

# Generate a range of angles to plot tuning curves
test_angles = np.linspace(-np.pi, np.pi, 100)
test_angles_tf = tf.constant(test_angles[np.newaxis, :, np.newaxis], dtype=tf.float32)

# Calculate tuning curves
with tf.Session() as sess:
    tuning_curves = sess.run(hd_ensemble.unnor_logpdf(test_angles_tf))[0]

# Plot tuning curves for a few example cells
example_hd_cells = [0, n_hd_cells // 3, 2 * n_hd_cells // 3]
hd_colors = ['c', 'm', 'y']
for i, cell_idx in enumerate(example_hd_cells):
    ax4.plot(test_angles, tuning_curves[:, cell_idx], color=hd_colors[i], label=f"Cell {cell_idx}")
    # Mark preferred angle
    preferred_angle = hd_ensemble.means[cell_idx]
    ax4.axvline(preferred_angle, color=hd_colors[i], linestyle='--')
ax4.legend()


# --- e) Activity of Example HD Cells over Time ---
ax5 = fig.add_subplot(2, 3, 5)
ax5.set_title("Activity of Example HD Cells")
ax5.set_xlabel("Time Step")
ax5.set_ylabel("Unnormalized Log Probability")
for i, cell_idx in enumerate(example_hd_cells):
    activity = hd_log_pdf[:, cell_idx]
    ax5.plot(activity, color=hd_colors[i], label=f"Cell {cell_idx}")
ax5.legend()

# --- f) Decoded Head Direction at a Specific Time ---
ax6 = fig.add_subplot(2, 3, 6, projection='polar')
ax6.set_title(f"Decoded Head Direction at t={time_idx}")

# The target is a softmax over HD cells.
hd_target_probs = hd_targets[time_idx]

# Plot the distribution on a polar plot
# We plot the probability of each cell at its preferred angle.
ax6.plot(hd_ensemble.means, hd_target_probs, 'o-', markersize=4, linewidth=1, label="Decoded Distribution")

# Mark true head direction
true_angle = angles[time_idx]
ax6.plot([0, true_angle], [0, np.max(hd_target_probs)], 'r-', linewidth=3, label="True Direction")
ax6.legend()

plt.tight_layout()
plt.show()