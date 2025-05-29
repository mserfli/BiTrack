from tracking.tracker import Tracker
from tracking.detections.detections import Detections
import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np
from utils import KittiObject3d
from tracking.trajectory_clustering_split_and_recombination import merge_forward_backward_trajectories
from plotly.subplots import make_subplots

# Constants
NUM_FRAMES = 400
SAMPLE_TIME = 0.1  # 100ms
AMPLITUDE = 5.0  # Amplitude of the cosine trajectory
FREQ_START = 0.025  # Starting frequency of the cosine trajectory
FREQ_END = 0.05  # Ending frequency of the cosine trajectory
VELOCITY_TARGET = np.array([2.0])  # Velocity of the target in x-direction
X_OFFSET_START = 10.0  # Starting x-offset for the target trajectory
Y_OFFSET_START = -2.0  # Starting y-offset for the ego vehicle
Y_OFFSET_END = 4.0  # Ending y-offset for the ego vehicle
EGO_POS_NOISE_STD = 0.025  # Standard deviation of noise for ego position
TGT_POS_NOISE_STD = 0.05  # Standard deviation of noise for target position

# Tracker configuration
tracker_config = {
    "t_miss": 10,
    "t_miss_new": 3,
    "t_hit": 2,
    "match_algorithm": "MCF",
    "aff_thresh": 0.5,
    "ang_thresh": 0.5,
    "app_thresh": 0.5,
    "ent_ex_score": 0.3,
    "offline": False,  # offline mode somehow does not smooth the trajectories
    "app_m": 0.9,
    "p": 5**2,  # TDOD still used equally for all dims
    "q": 0.5,  # TDOD still used equally for all dims
    "r": 1,  # TDOD still used equally for all dims
    "ang_vel": True,
    "vel_reinit": True,
    "sim_metric": "NCD",
    "T": SAMPLE_TIME,
}


def generate_target_trajectory() -> np.ndarray:
    """
    Generates a target trajectory with linear motion in the x-direction,
    linearly increasing frequency, and cosine motion in the y-direction.

    Returns:
        np.ndarray: Target trajectory as an array of shape (NUM_FRAMES, 2).
    """
    frames = np.arange(NUM_FRAMES) * SAMPLE_TIME
    # Linear motion in x-direction with offset
    x_positions = frames * VELOCITY_TARGET[0] + X_OFFSET_START
    # Linearly increasing frequency
    frequencies = np.linspace(FREQ_START, FREQ_END, NUM_FRAMES)
    # Cosine motion in y-direction
    y_positions = AMPLITUDE * np.cos(2 * np.pi * frequencies * frames)
    return np.column_stack((x_positions, y_positions))


def generate_ego_trajectory() -> np.ndarray:
    """
    Generates a reference trajectory for the ego vehicle with straight diagonal motion.

    Returns:
        np.ndarray: Ego trajectory as an array of shape (NUM_FRAMES, 2).
    """
    frames = np.arange(NUM_FRAMES) * SAMPLE_TIME
    x_positions = frames * VELOCITY_TARGET[0]  # Linear motion in x-direction
    # Linear motion in y-direction
    y_positions = np.linspace(Y_OFFSET_START, Y_OFFSET_END, NUM_FRAMES)
    return np.column_stack((x_positions, y_positions))


def create_detections_from_trajectories(
    ego_trajectory: np.ndarray,
    target_trajectory: np.ndarray,
    ego_box_shape: np.ndarray = np.array([1.0, 1.0, 1.0]),
    ego_shape_noise_std: float = 0.0,
    ego_position_noise_std: float = 0.0,
    target_box_shape: np.ndarray = np.array([1.0, 1.0, 1.0]),
    target_shape_noise_std: float = 0.0,
    target_position_noise_std: float = 0.0,
) -> list[Detections]:
    """
    Creates detections from globally defined trajectories for tracking.

    Args:
        ego_trajectory (np.ndarray): Ego vehicle trajectory (NUM_FRAMES, 2).
        target_trajectory (np.ndarray): Target trajectory (NUM_FRAMES, 2).
        ego_box_shape (np.ndarray): Shape of the ego bounding box [length, width, height].
        ego_shape_noise_std (float): Standard deviation of noise applied to the ego box shape.
        ego_position_noise_std (float): Standard deviation of noise applied to the ego position.
        target_box_shape (np.ndarray): Shape of the target bounding box [length, width, height].
        target_shape_noise_std (float): Standard deviation of noise applied to the target box shape.
        target_position_noise_std (float): Standard deviation of noise applied to the target position.

    Returns:
        list[Detections]: List of detections for each frame.
    """
    detections_per_frame = []

    for frame_idx in range(len(ego_trajectory)):
        # Get ego and target positions for the current frame
        ego_position = ego_trajectory[frame_idx] + \
            np.random.normal(0, ego_position_noise_std, size=2)
        target_position = target_trajectory[frame_idx] + \
            np.random.normal(0, target_position_noise_std, size=2)

        # Apply noise to the box shapes
        ego_box_shape_noisy = ego_box_shape + \
            np.random.normal(0, ego_shape_noise_std, size=3)
        target_box_shape_noisy = target_box_shape + \
            np.random.normal(0, target_shape_noise_std, size=3)

        # Create 3D bounding boxes for ego and target
        # [x, y, z, l, w, h, yaw]
        ego_box = np.array([*ego_position, 0.0, *ego_box_shape_noisy, 0.0])
        target_box = np.array(
            [*target_position, 0.0, *target_box_shape_noisy, 0.0])

        # Combine ego and target boxes into a single array
        boxes3d = np.stack([ego_box, target_box])

        # Create dummy objects for ego and target
        objs = [
            KittiObject3d(),  # Ego object
            KittiObject3d(),  # Target object
        ]

        # Set dummy tracking scores and sample IDs for ego and target
        for obj, sample_id in zip(objs, [frame_idx, frame_idx]):
            obj.tracking_score = 1.0
            obj.sample_id = sample_id  # Add sample_id attribute

        # Create a Detections object for the current frame
        detections = Detections(
            boxes3d=boxes3d,
            objs=objs,
            similarity=None,  # No similarity data for now
            embeds=None,      # No embeddings for now
            coor_2d_inds=None  # No 2D coordinates for now
        )

        # Append detections for the current frame
        detections_per_frame.append(detections)

    return detections_per_frame


def apply_tracker(
    detections: list[Detections], tracker_config: dict, backward: bool = False
) -> dict[int, tuple[list[np.ndarray], list[KittiObject3d]], list[np.ndarray]]:
    """
    Applies the tracker to the detections to create forward or backward tracks.

    Args:
        detections (list[Detections]): List of detections for each frame.
        tracker_config (dict): Tracker configuration.
        backward (bool): Whether to perform backward tracking.

    Returns:
        dict[int, tuple[list[np.ndarray], list[KittiObject3d], list[np.ndarray]]]: Trajectories generated by the tracker.
    """
    # Initialize the tracker
    tracker = Tracker(**tracker_config)
    tracker.reset()
    trajectories = {}
    last_frame = None

    # Iterate through frames (forward or backward)
    frame_indices = reversed(range(len(detections))
                             ) if backward else range(len(detections))
    for frame_idx in frame_indices:
        current_detections = detections[frame_idx]

        # Determine the number of passed frames
        if last_frame is None:
            last_frame = frame_idx
            num_passed_frames = 1
        else:
            num_passed_frames = abs(frame_idx - last_frame)
            last_frame = frame_idx

        # Perform tracking for the current frame
        _, pred_boxes = tracker.predict(num_passed_frames)
        matched, entry_dets, exit_trks, false_trks = tracker.associate(
            pred_boxes, current_detections)
        tracker.update(matched, entry_dets, exit_trks,
                       false_trks, current_detections)
        online_tracks, dead_tracks = tracker.track_management()

        # Collect tracking data
        if tracker.offline:
            # Offline mode: Collect data from dead tracks
            for trk in dead_tracks:
                if trk.max_hits >= tracker.t_hit:
                    for obj in trk.objs:
                        obj.tracking_id = trk.id
                    trajectories[trk.id] = (
                        [trk.boxes[::-1], trk.objs[::-1],
                            np.sqrt(np.diag(trk.filter.kf.P[::-1]))[:, np.newaxis] if backward else trk.objs]
                        if backward
                        else [trk.boxes, trk.objs, np.sqrt(np.diag(trk.filter.kf.P))[:, np.newaxis]]
                    )
        else:
            # Online mode: Collect data from online tracks
            for trk in online_tracks:
                trk.obj.tracking_id = trk.id
                if trk.id not in trajectories:
                    trajectories[trk.id] = ([], [], [])
                trajectories[trk.id][0].append(
                    trk.filter.x)  # Append the latest box
                # Append the latest object
                trajectories[trk.id][1].append(trk.obj)
                trajectories[trk.id][2].append(
                    np.sqrt(np.diag(trk.filter.kf.P))[:, np.newaxis])  # Append the latest box

    # Add remaining tracks to trajectories in offline mode
    if tracker.offline:
        for trk in tracker.tracks:
            if trk.max_hits >= tracker.t_hit:
                for obj in trk.objs:
                    obj.tracking_id = trk.id
                trajectories[trk.id] = (
                    [trk.boxes[::-1], trk.objs[::-1],
                        np.sqrt(np.diag(trk.filter.kf.P[::-1]))[:, np.newaxis] if backward else trk.objs]
                    if backward
                    else [trk.boxes, trk.objs, np.sqrt(np.diag(trk.filter.kf.P))[:, np.newaxis]]
                )

    return trajectories


# Generate trajectories
target_trajectory = generate_target_trajectory()
ego_trajectory = generate_ego_trajectory()

detections = create_detections_from_trajectories(
    ego_trajectory,
    target_trajectory,
    ego_box_shape=np.array([1.0, 1.0, 1.0]),
    ego_shape_noise_std=0.1,
    ego_position_noise_std=EGO_POS_NOISE_STD,
    target_box_shape=np.array([1.0, 1.0, 1.0]),
    target_shape_noise_std=0.1,
    target_position_noise_std=TGT_POS_NOISE_STD
)

# Apply tracker for forward and backward passes
forward_tracks = apply_tracker(detections, tracker_config, backward=False)
backward_tracks = apply_tracker(detections, tracker_config, backward=True)

# Reverse backward tracks to align with forward tracks
aligned_backward_tracks = {
    track_id: (boxes[::-1], objs[::-1], stdev[::-1])
    for track_id, (boxes, objs, stdev) in backward_tracks.items()
}

# Merge forward and backward trajectories using the existing method
merged_tracks = merge_forward_backward_trajectories(
    forward_tracks, aligned_backward_tracks)

# ========= visualization =========

# Create Plotly figure
fig_course = go.Figure()

# Plot target trajectory
fig_course.add_trace(go.Scatter(
    x=target_trajectory[:, 0], y=target_trajectory[:, 1],
    mode='lines',
    name='Target Trajectory',
    line=dict(color='blue', width=2)
))

# Plot ego vehicle trajectory
fig_course.add_trace(go.Scatter(
    x=ego_trajectory[:, 0], y=ego_trajectory[:, 1],
    mode='lines',
    name='Ego Vehicle Trajectory',
    line=dict(color='red', width=2)
))

# Extract detected positions from detections
detected_ego_positions = np.array([d.boxes3d[0][:2] for d in detections])
detected_target_positions = np.array([d.boxes3d[1][:2] for d in detections])

# Plot detected ego positions
fig_course.add_trace(go.Scatter(
    x=detected_ego_positions[:, 0], y=detected_ego_positions[:, 1],
    mode='markers',
    name='Detected Ego Positions',
    marker=dict(color='red', size=6, symbol='circle')
))

# Plot detected target positions
fig_course.add_trace(go.Scatter(
    x=detected_target_positions[:, 0], y=detected_target_positions[:, 1],
    mode='markers',
    name='Detected Target Positions',
    marker=dict(color='blue', size=6, symbol='circle')
))

# Define colors for legend groups
forward_color = "green"
backward_color = "purple"
merged_color = "orange"

# Plot forward tracks
for idx, (track_id, (boxes, _, _)) in enumerate(forward_tracks.items()):
    track_positions = np.array([box[:2] for box in boxes]).reshape(-1, 2)
    fig_course.add_trace(go.Scatter(
        x=track_positions[:, 0], y=track_positions[:, 1],
        mode='lines+markers',
        name=f'Forward Track {track_id}',
        # Use the same color for all forward tracks
        line=dict(color=forward_color, width=2),
        marker=dict(size=6),
        legendgroup="Forward Tracks",  # Group forward tracks
        text=[f"Index: {i}, Position: ({x:.2f}, {y:.2f})" for i, (x, y) in enumerate(
            track_positions)],  # Tooltip text
        hoverinfo="text"  # Display only the text in the tooltip
    ))

# Plot backward tracks
for idx, (track_id, (boxes, _, _)) in enumerate(backward_tracks.items()):
    track_positions = np.array([box[:2] for box in boxes]).reshape(-1, 2)
    fig_course.add_trace(go.Scatter(
        x=track_positions[:, 0], y=track_positions[:, 1],
        mode='lines+markers',
        name=f'Backward Track {track_id}',
        # Use the same color for all backward tracks
        line=dict(color=backward_color, width=2),
        marker=dict(size=6),
        legendgroup="Backward Tracks",  # Group backward tracks
        text=[f"Index: {i}, Position: ({x:.2f}, {y:.2f})" for i, (x, y) in enumerate(
            track_positions)],  # Tooltip text
        hoverinfo="text"  # Display only the text in the tooltip
    ))

# Plot merged tracks
for idx, (track_id, (boxes, _)) in enumerate(merged_tracks.items()):
    track_positions = np.array([box[:2] for box in boxes]).reshape(-1, 2)
    fig_course.add_trace(go.Scatter(
        x=track_positions[:, 0], y=track_positions[:, 1],
        mode='lines+markers',
        name=f'Merged Track {track_id}',
        # Use the same color for all merged tracks
        line=dict(color=merged_color, width=2),
        marker=dict(size=6),
        legendgroup="Merged Tracks",  # Group merged tracks
        text=[f"Index: {i}, Position: ({x:.2f}, {y:.2f})" for i, (x, y) in enumerate(
            track_positions)],  # Tooltip text
        hoverinfo="text"  # Display only the text in the tooltip
    ))

# Update layout
fig_course.update_layout(
    title="Merged Target and Ego Vehicle Trajectories",
    xaxis_title="X Position",
    yaxis_title="Y Position",
    legend_title="Trajectories",
    template="plotly_white"
)

# Show the plot
fig_course.show()

# Create a new figure with subplots
fig_tracks = make_subplots(rows=4, cols=1, shared_xaxes=True,
                           subplot_titles=["X Position", "Y Position", "VX Velocity", "VY Velocity"])

# Define colors for tracks
track_colors = pc.qualitative.Plotly[:len(
    forward_tracks) + len(backward_tracks)]


def add_trace_to_subplot(fig, track_positions, track_stdev, track_id, track_type, row, col, idx, track_colors, variable_name):
    fig.add_trace(go.Scatter(
        x=np.arange(len(track_positions)),
        y=track_positions[:, row - 1],  # Extract the variable (X, Y, VX, VY)
        mode='lines',
        name=f'{track_type} Track {track_id} ({variable_name})',
        line=dict(color=track_colors[idx % len(track_colors)]),
        legendgroup=f'{track_type} Track {track_id}',
        text=[f"Index: {i}, val: {val:.2f}" for i, val in enumerate(
            track_positions[:, row - 1])],  # Tooltip text
        hoverinfo="text"  # Display only the text in the tooltip
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=np.arange(len(track_positions)),
        # Extract the variable (X, Y, VX, VY)
        y=track_positions[:, row - 1] + track_stdev[:, row - 1],
        mode='lines',
        name=f'{track_type} TrackStdDev {track_id} ({variable_name})',
        line=dict(color=track_colors[idx % len(track_colors)], dash='dash'),
        legendgroup=f'{track_type} TrackStdDev {track_id}',
    ), row=row, col=col)


# Define variable names for subplots
variable_names = ["X", "Y", "VX", "VY"]

# Plot forward tracks
for idx, (track_id, (boxes, _, stddevs)) in enumerate(forward_tracks.items()):
    track_positions = np.array(
        [box[[0, 1, 7, 8], :] for box in boxes]).reshape(-1, 4)  # Extract x, y, vx, vy
    track_stddev = np.array(
        [stddev[[0, 1, 7, 8], :] for stddev in stddevs]).reshape(-1, 4)  # Extract x, y, vx, vy
    for row, variable_name in enumerate(variable_names, start=1):
        add_trace_to_subplot(fig_tracks, track_positions, track_stddev, track_id,
                             "Forward", row, 1, idx, track_colors, variable_name)

# Plot backward tracks
for idx, (track_id, (boxes, _, stddevs)) in enumerate(aligned_backward_tracks.items()):
    track_positions = np.array(
        [box[[0, 1, 7, 8], :] for box in boxes]).reshape(-1, 4)  # Extract x, y, vx, vy
    track_stddev = np.array(
        [stddev[[0, 1, 7, 8], :] for stddev in stddevs]).reshape(-1, 4)  # Extract x, y, vx, vy
    for row, variable_name in enumerate(variable_names, start=1):
        add_trace_to_subplot(fig_tracks, track_positions, track_stddev, track_id,
                             "Backward", row, 1, idx, track_colors[len(forward_tracks):], variable_name)

# Update layout
fig_tracks.update_layout(
    title="Tracks (X, Y, VX, VY) with State Uncertainty",
    xaxis_title="Frame Index",
    yaxis_title="Value",
    height=1000,
    template="plotly_white"
)

# Show the figure
fig_tracks.show()
