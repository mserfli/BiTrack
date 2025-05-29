from typing import Dict, List, Tuple

import numpy as np

from utils import KittiTrack3d, visualize_trajectories


def get_overlaps_of_trajectories(
    a_trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
    b_trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
) -> Tuple[List[int], List[int], List[List[Tuple[int, int]]]]:
    """
    Computes overlaps between two sets of trajectories based on frame IDs and object locations.

    Args:
        a_trajectories (Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]]): 
            A dictionary where keys are trajectory IDs and values are tuples containing:
            - A list of bounding boxes (as numpy arrays).
            - A list of `KittiTrack3d` objects representing tracked objects.
        b_trajectories (Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]]): 
            Another dictionary with the same structure as `a_trajectories`.

    Returns:
        Tuple[List[int], List[int], List[List[Tuple[int, int]]]]:
            - `a_ids`: List of trajectory IDs from `a_trajectories`.
            - `b_ids`: List of trajectory IDs from `b_trajectories`.
            - `overlap_pair_inds`: A 2D list where `overlap_pair_inds[i][j]` contains pairs of indices `(a_i, b_j)` 
              indicating overlapping frames between trajectory `a_ids[i]` and `b_ids[j]`.

    Raises:
        AssertionError: If the frames in any trajectory are not sorted in ascending order.

    Notes:
        - The function assumes that the trajectories are ordered by frame IDs.
        - Overlaps are determined by matching frame IDs and verifying that the object locations are identical.
        - A two-pointer search algorithm is used to efficiently find overlaps between trajectories.

    Example:
        Given two trajectories:
        - `a_trajectories = {1: ([box1, box2], [obj1, obj2])}`
        - `b_trajectories = {2: ([box3, box4], [obj3, obj4])}`
        The function will return:
        - `a_ids = [1]`
        - `b_ids = [2]`
        - `overlap_pair_inds = [[[(0, 0), (1, 1)]]]` (if frames and locations match).
    """
    n = len(a_trajectories)
    m = len(b_trajectories)
    overlap_pair_inds = [[[] for _ in range(m)] for _ in range(n)]
    a_ids = list(a_trajectories.keys())
    b_ids = list(b_trajectories.keys())
    for i, a_id in enumerate(a_ids):
        _, a_objs = a_trajectories[a_id]
        a_frames = [a_obj.sample_id for a_obj in a_objs]
        # check frames are sorted
        assert all(
            a_frames[idx] < a_frames[idx + 1] for idx in range(len(a_frames) - 1)
        ), str(a_frames)

        for j, b_id in enumerate(b_ids):
            _, b_objs = b_trajectories[b_id]
            b_frames = [b_obj.sample_id for b_obj in b_objs]
            # check frames are sorted
            assert all(
                b_frames[idx] < b_frames[idx + 1] for idx in range(len(b_frames) - 1)
            ), str(b_frames)

            # two-pointers search
            a_i = 0
            b_j = 0
            while a_i < len(a_frames) and b_j < len(b_frames):
                if a_frames[a_i] == b_frames[b_j]:
                    if np.array_equal(a_objs[a_i].loc, b_objs[b_j].loc):
                        overlap_pair_inds[i][j].append((a_i, b_j))
                        # TODO its very unlikely that the locations of an object from forward and backward tracking is identical
                        # due to noise and observation length
                        # TODO why not calculating the mahalanobis distance between the two locations?
                    a_i += 1
                    b_j += 1
                else:
                    if a_frames[a_i] > b_frames[b_j]:
                        b_j += 1
                    else:
                        a_i += 1
    return a_ids, b_ids, overlap_pair_inds


def get_clusters(overlap_arr) -> List[Tuple[List[int]]]:
    """
    Groups overlapping trajectories into clusters based on their connections.

    Args:
        overlap_arr (List[List[bool]]): 
            A 2D boolean matrix where `overlap_arr[i][j]` is `True` if trajectory `a_ids[i]` overlaps with trajectory `b_ids[j]`.

    Returns:
        List[Tuple[List[int], List[int]]]: 
            A list of clusters, where each cluster is represented as a tuple:
            - `cluster_a`: A list of indices corresponding to trajectories in `a_trajectories`.
            - `cluster_b`: A list of indices corresponding to trajectories in `b_trajectories`.

    Notes:
        - The function uses a breadth-first search-like approach to traverse connections and form clusters.
        - It alternates between `a_trajectories` and `b_trajectories` while traversing connections to ensure all overlapping trajectories are included in the cluster.
        - Unprocessed indices are tracked using sets (`a_idx_set` and `b_idx_set`), which are updated as indices are added to clusters.

    Example:
        Given the following `overlap_arr`:
        ```
        overlap_arr = [
            [True, False, True],
            [False, True, False],
            [True, False, False]
        ]
        ```
        The function will return:
        ```
        clusters = [
            ([0, 2], [0]),  # Trajectories 0 and 2 from [a_trajectories](http://_vscodecontentref_/1) overlap with trajectory 0 from [b_trajectories](http://_vscodecontentref_/2).
            ([1], [1]),     # Trajectory 1 from [a_trajectories](http://_vscodecontentref_/3) overlaps with trajectory 1 from [b_trajectories](http://_vscodecontentref_/4).
            ([0], [2])      # Trajectory 0 from [a_trajectories](http://_vscodecontentref_/5) overlaps with trajectory 2 from [b_trajectories](http://_vscodecontentref_/6).
        ]
        ```
    """
    n = len(overlap_arr)
    m = len(overlap_arr[0])
    a_idx_set = set(range(n))
    b_idx_set = set(range(m))
    clusters = []
    while a_idx_set or b_idx_set:
        cluster_a = []
        cluster_b = []
        if a_idx_set:
            cur_idx = a_idx_set.pop()
            cluster_a.append(cur_idx)
            cur_inds = [
                j for j in range(m) if overlap_arr[cur_idx][j] and j in b_idx_set
            ]
            is_a = True
        else:
            cur_idx = b_idx_set.pop()
            cluster_b.append(cur_idx)
            cur_inds = [
                i for i in range(n) if overlap_arr[i][cur_idx] and i in a_idx_set
            ]
            is_a = False
        while len(cur_inds) > 0:
            next_inds = []
            for x in cur_inds:
                if is_a:
                    b_idx_set.remove(x)
                    cluster_b.append(x)
                    next_inds += [
                        i for i in range(n) if overlap_arr[i][x] and i in a_idx_set
                    ]
                else:
                    a_idx_set.remove(x)
                    cluster_a.append(x)
                    next_inds += [
                        j for j in range(m) if overlap_arr[x][j] and j in b_idx_set
                    ]
            cur_inds = set(next_inds)
            is_a = not is_a
        clusters.append((cluster_a, cluster_b))
    return clusters


def group_consecutive_inds_dict(inds_dict: Dict[int, List[int]]):
    # this is inplace modification
    for key, indices in inds_dict.items():
        consecutive_inds = []
        last_idx = None
        for x in indices:
            if last_idx is None:
                last_idx = x
                consecutive_inds.append([x])
            else:
                if x == last_idx + 1:
                    consecutive_inds[-1].append(x)
                else:
                    consecutive_inds.append([x])
        inds_dict[key] = consecutive_inds


def merge_distinct(
    common_inds_dict,
    ids,
    trajectories,
    cluster_distinct_inds,
    all_boxes,
    all_objs,
    all_frames,
):
    for idx, common_inds_list in common_inds_dict.items():
        boxes, objs = trajectories[ids[idx]]
        distinct_inds_list = cluster_distinct_inds[idx]
        valid_mask = [False] * len(distinct_inds_list)
        common_inds_list.sort()
        i = 0
        j = 0
        while i < len(common_inds_list) and j < len(distinct_inds_list):
            if distinct_inds_list[j][0] == 0:
                if common_inds_list[i][0] == distinct_inds_list[j][-1] + 1:
                    valid_mask[j] = True
                    i += 1
                    j += 1
                elif common_inds_list[i][0] < distinct_inds_list[j][-1] + 1:
                    i += 1
                else:
                    j += 1
            else:
                if common_inds_list[i][-1] == distinct_inds_list[j][0] - 1:
                    valid_mask[j] = True
                    i += 1
                    j += 1
                elif common_inds_list[i][-1] < distinct_inds_list[j][0] - 1:
                    i += 1
                else:
                    j += 1

        distinct_inds_list_new = []
        for x in range(len(distinct_inds_list)):
            if valid_mask[x]:
                for distinct_idx in distinct_inds_list[x]:
                    if objs[distinct_idx].sample_id not in all_frames:
                        all_boxes.append(boxes[distinct_idx])
                        all_objs.append(objs[distinct_idx])
                        all_frames.append(objs[distinct_idx].sample_id)
            else:
                distinct_inds_list_new.append(distinct_inds_list[x])
        cluster_distinct_inds[idx] = distinct_inds_list_new


def use_forward_common_trajectories(
    a_trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
    b_trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
):
    if len(a_trajectories) == 0 or len(b_trajectories) == 0:
        return a_trajectories if len(a_trajectories) > 0 else b_trajectories
    a_ids, b_ids, overlap_pair_inds = get_overlaps_of_trajectories(
        a_trajectories, b_trajectories
    )
    for i, a_id in enumerate(a_ids):
        boxes, objs = a_trajectories[a_id]
        inds = [[pair[0] for pair in pair_inds]
                for pair_inds in overlap_pair_inds[i]]
        inds = sum(inds, [])
        boxes = [boxes[x] for x in inds]
        objs = [objs[x] for x in inds]
        a_trajectories[a_id] = (boxes, objs)
    return a_trajectories


def merge_forward_backward_trajectories(
    a_trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
    b_trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
    visualize_contradictions=False,
    merge_common_only=True,
) -> Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]]:
    """
    Merges two sets of trajectories obtained from forward and backward tracking, resolving overlaps and contradictions.

    Args:
        a_trajectories (Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]]): 
            A dictionary where keys are trajectory IDs and values are tuples containing:
            - A list of bounding boxes (as numpy arrays).
            - A list of `KittiTrack3d` objects representing tracked objects.
        b_trajectories (Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]]): 
            Another dictionary with the same structure as `a_trajectories`.
        visualize_contradictions (bool, optional): 
            If `True`, visualizes trajectories with contradictions. Defaults to `False`.
        merge_common_only (bool, optional): 
            If `True`, merges only the overlapping parts of trajectories. Defaults to `True`.

    Returns:
        Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]]: 
            A dictionary containing merged trajectories, where:
            - Keys are new trajectory IDs.
            - Values are tuples of bounding boxes and tracked objects.

    Raises:
        AssertionError: If the frames in any trajectory are not sorted in ascending order.

    Notes:
        - The function uses `get_overlaps_of_trajectories` to identify overlaps between trajectories.
        - Trajectories are grouped into clusters based on overlaps using `get_clusters`.
        - Overlapping and distinct parts of trajectories are merged based on the `merge_common_only` flag.
        - Contradictions in trajectories can be visualized if `visualize_contradictions` is enabled.

    Example:
        Given two sets of trajectories:
        - `a_trajectories = {1: ([box1, box2], [obj1, obj2])}`
        - `b_trajectories = {2: ([box3, box4], [obj3, obj4])}`
        The function will merge overlapping trajectories and return a new dictionary of merged trajectories.
    """
    if len(a_trajectories) == 0 or len(b_trajectories) == 0:
        return a_trajectories if len(a_trajectories) > 0 else b_trajectories
    a_ids, b_ids, overlap_pair_inds = get_overlaps_of_trajectories(
        a_trajectories, b_trajectories
    )
    # Each element is_overlapped[i][j] is True if there are overlapping frames between trajectory a_ids[i] from a_trajectories
    # and trajectory b_ids[j] from b_trajectories. Otherwise, it is False.
    is_overlapped = [
        [len(overlap_pair_inds[i][j]) > 0 for j in range(len(b_ids))]
        for i in range(len(a_ids))
    ]
    # transform 2d boolean array to 2d list of indices
    clusters = get_clusters(is_overlapped)
    # merge
    merged_trajectories = {}
    cur_id = 1
    for cluster in clusters:
        cluster_a_inds: List[int] = cluster[0]
        cluster_b_inds: List[int] = cluster[1]

        cluster_common_objs = []
        cluster_common_boxes = []
        cluster_a_common_inds = {x: [] for x in cluster_a_inds}
        cluster_b_common_inds = {x: [] for x in cluster_b_inds}
        for a_idx in cluster_a_inds:
            a_boxes, a_objs = a_trajectories[a_ids[a_idx]]
            for b_idx in cluster_b_inds:
                for a_common_idx, b_common_idx in overlap_pair_inds[a_idx][b_idx]:
                    cluster_common_objs.append(a_objs[a_common_idx])
                    cluster_common_boxes.append(a_boxes[a_common_idx])
                    cluster_a_common_inds[a_idx].append(a_common_idx)
                    cluster_b_common_inds[b_idx].append(b_common_idx)

        if merge_common_only:
            # based on the metric of get_overlaps_of_trajectories only the common objects from forward and backward
            # trajectories are merged
            cluster_common_frames = [x.sample_id for x in cluster_common_objs]
            if len(set(cluster_common_frames)) == len(cluster_common_frames):
                if len(cluster_common_objs) > 0:
                    for x in cluster_common_objs:
                        x.tracking_id = cur_id  # Assign a new unique tracking ID to all objects in the cluster
                    merged_trajectories[cur_id] = (
                        cluster_common_boxes,
                        cluster_common_objs,
                    )
                    cur_id += 1
                continue

        else:
            # Initialize lists to store distinct boxes and objects for both clusters
            cluster_a_distinct_boxes = []
            cluster_a_distinct_objs = []
            cluster_b_distinct_boxes = []
            cluster_b_distinct_objs = []

            # Process distinct indices for trajectories in cluster_a
            for a_idx in cluster_a_inds:
                a_boxes, a_objs = a_trajectories[a_ids[a_idx]]
                # Identify indices that are not part of the common indices
                a_distinct_inds = [
                    x for x in range(len(a_objs)) if x not in cluster_a_common_inds[a_idx]
                ]
                # Collect distinct boxes and objects
                cluster_a_distinct_boxes.extend(
                    [a_boxes[x] for x in a_distinct_inds])
                cluster_a_distinct_objs.extend(
                    [a_objs[x] for x in a_distinct_inds])

            # Process distinct indices for trajectories in cluster_b
            for b_idx in cluster_b_inds:
                b_boxes, b_objs = b_trajectories[b_ids[b_idx]]
                # Identify indices that are not part of the common indices
                b_distinct_inds = [
                    x for x in range(len(b_objs)) if x not in cluster_b_common_inds[b_idx]
                ]
                # Collect distinct boxes and objects
                cluster_b_distinct_boxes.extend(
                    [b_boxes[x] for x in b_distinct_inds])
                cluster_b_distinct_objs.extend(
                    [b_objs[x] for x in b_distinct_inds])

            # Combine all boxes and objects from common and distinct parts
            cluster_all_boxes = (
                cluster_common_boxes + cluster_a_distinct_boxes + cluster_b_distinct_boxes
            )
            cluster_all_objs = (
                cluster_common_objs + cluster_a_distinct_objs + cluster_b_distinct_objs
            )
            cluster_all_frames = [x.sample_id for x in cluster_all_objs]

            # Ensure there are no duplicate frames in the merged cluster
            if len(set(cluster_all_frames)) == len(cluster_all_frames):
                if len(cluster_all_objs) > 0:
                    # Assign a unique tracking ID to all objects in the cluster
                    for x in cluster_all_objs:
                        x.tracking_id = cur_id
                    # Add the merged cluster to the trajectories dictionary
                    merged_trajectories[cur_id] = (
                        cluster_all_boxes, cluster_all_objs)
                    cur_id += 1
                continue

        if visualize_contradictions:
            cluster_a_boxes = []
            for a_idx in cluster_a_inds:
                a_boxes, _ = a_trajectories[a_ids[a_idx]]
                cluster_a_boxes.append(a_boxes)
            visualize_trajectories(cluster_a_boxes)
            cluster_b_boxes = []
            for b_idx in cluster_b_inds:
                b_boxes, _ = b_trajectories[b_ids[b_idx]]
                cluster_b_boxes.append(b_boxes[::-1])  # reverse direction
            visualize_trajectories(cluster_b_boxes)

        # Initialize the list to store overlapping tracklets between trajectories in cluster_a_inds and cluster_b_inds.
        # Each tracklet represents a sequence of consecutive overlapping frames.
        common_tracklets = []
        for a_idx in cluster_a_inds:
            for b_idx in cluster_b_inds:
                # Variables to track the last indices of overlapping frames for both trajectories.
                last_a_idx = None
                last_b_idx = None
                # Temporary storage for the current tracklet.
                cur_tracklet = []
                # Extract bounding boxes and objects for the backward trajectory.
                b_boxes, b_objs = b_trajectories[b_ids[b_idx]]
                b_len = len(b_boxes)  # Length of the backward trajectory.

                # Iterate over overlapping frame indices between the current pair of trajectories.
                for a_common_idx, b_common_idx in overlap_pair_inds[a_idx][b_idx]:
                    # Get the frame ID of the overlapping object.
                    cur_frame = b_objs[b_common_idx].sample_id
                    # Reverse indices for backward tracking.
                    b_common_idx = b_len - 1 - b_common_idx

                    if last_a_idx is None:
                        # If this is the first overlapping frame, initialize a new tracklet.
                        cur_tracklet = [
                            # Index of the trajectory in a_trajectories.
                            a_idx,
                            # Index of the trajectory in b_trajectories.
                            b_idx,
                            # Last index of overlapping frame in a_objs.
                            a_common_idx,
                            # Last index of overlapping frame in b_objs (reversed).
                            b_common_idx,
                            # Length of the tracklet (number of consecutive overlapping frames).
                            1,
                            [cur_frame],  # List of frame IDs for the tracklet.
                        ]
                    else:
                        if (
                            a_common_idx == last_a_idx + 1
                            and b_common_idx == last_b_idx - 1
                        ):
                            # If the current indices are consecutive, extend the current tracklet.
                            # Update the last index for a_objs.
                            cur_tracklet[2] = a_common_idx
                            # Increment the tracklet length.
                            cur_tracklet[4] += 1
                            # Add the current frame ID to the tracklet.
                            cur_tracklet[5].append(cur_frame)
                        else:
                            # If the indices are not consecutive, finalize the current tracklet and start a new one.
                            common_tracklets.append(cur_tracklet)
                            cur_tracklet = [
                                a_idx,
                                b_idx,
                                a_common_idx,
                                b_common_idx,
                                1,
                                [cur_frame],
                            ]
                    # Update the last indices for the next iteration.
                    last_a_idx = a_common_idx
                    last_b_idx = b_common_idx

                # If there is an unfinished tracklet, add it to the list.
                if len(cur_tracklet) > 0:
                    common_tracklets.append(cur_tracklet)

        # Sort tracklets by their last indices for both a_trajectories and b_trajectories.
        a_sorted_tracklets = {a_idx: [] for a_idx in cluster_a_inds}
        b_sorted_tracklets = {b_idx: [] for b_idx in cluster_b_inds}
        for i, tracklet in enumerate(common_tracklets):
            a_idx, b_idx, a_last_idx, b_last_idx, _, _ = tracklet
            a_sorted_tracklets[a_idx].append((a_last_idx, i))
            b_sorted_tracklets[b_idx].append((b_last_idx, i))

        # Sort the tracklets within each trajectory by their last indices.
        for x in a_sorted_tracklets.values():
            x.sort()
        for x in b_sorted_tracklets.values():
            x.sort()

        # Convert sorted tracklets into lists of indices for easier access.
        a_sorted_tracklets = {k: [x[1] for x in v]
                              for k, v in a_sorted_tracklets.items()}
        b_sorted_tracklets = {k: [x[1] for x in v]
                              for k, v in b_sorted_tracklets.items()}

        # Initialize scores and choices for connecting tracklets.
        # Scores for connections before the current tracklet.
        before_scores = [-1] * len(common_tracklets)
        # Scores for connections after the current tracklet.
        after_scores = [-1] * len(common_tracklets)
        # Choices for connections before the current tracklet.
        before_choices = [-1] * len(common_tracklets)
        # Choices for connections after the current tracklet.
        after_choices = [-1] * len(common_tracklets)

        # Scores for alternative (bad) connections before the current tracklet.
        before_bad_scores = [-1] * len(common_tracklets)
        # Scores for alternative (bad) connections after the current tracklet.
        after_bad_scores = [-1] * len(common_tracklets)
        # Choices for alternative (bad) connections before the current tracklet.
        before_bad_choices = [-1] * len(common_tracklets)
        # Choices for alternative (bad) connections after the current tracklet.
        after_bad_choices = [-1] * len(common_tracklets)

        # Iterate over each tracklet to calculate scores and determine connections.
        for i, tracklet in enumerate(common_tracklets):
            a_idx, b_idx, a_last_idx, b_last_idx, _, _ = tracklet
            cur_a_tracklets: list = a_sorted_tracklets[a_idx]
            cur_b_tracklets: list = b_sorted_tracklets[b_idx]

            # Determine the position of the current tracklet in the sorted lists.
            a_k = cur_a_tracklets.index(i)
            b_k = cur_b_tracklets.index(i)

            # Calculate scores for connections before the current tracklet.
            if a_k == 0:
                a_before_score = -1
                a_before_tracklet_idx = "placeholder"
            else:
                a_before_tracklet_idx = cur_a_tracklets[a_k - 1]
                a_before_score = common_tracklets[a_before_tracklet_idx][2]

            if b_k < len(cur_b_tracklets) - 1:
                b_before_tracklet_idx = cur_b_tracklets[b_k + 1]
                b_before_score = common_tracklets[b_before_tracklet_idx][3]
            else:
                b_before_score = -1
                b_before_tracklet_idx = "placeholder"

            # Calculate scores for connections after the current tracklet.
            if a_k < len(cur_a_tracklets) - 1:
                a_after_tracklet_idx = cur_a_tracklets[a_k + 1]
                a_after_score = common_tracklets[a_after_tracklet_idx][2]
            else:
                a_after_score = -1
                a_after_tracklet_idx = "placeholder"

            if b_k == 0:
                b_after_score = -1
                b_after_tracklet_idx = "placeholder"
            else:
                b_after_tracklet_idx = cur_b_tracklets[b_k - 1]
                b_after_score = common_tracklets[b_after_tracklet_idx][3]

            # Compare scores and update choices for connections before and after the current tracklet.
            if a_before_score >= b_before_score:
                before_scores[i] = a_before_score
                before_choices[i] = a_before_tracklet_idx
                before_bad_scores[i] = b_before_score
                before_bad_choices[i] = b_before_tracklet_idx
            else:
                before_scores[i] = b_before_score
                before_choices[i] = b_before_tracklet_idx
                before_bad_scores[i] = a_before_score
                before_bad_choices[i] = a_before_tracklet_idx

            if a_after_score >= b_after_score:
                after_scores[i] = a_after_score
                after_choices[i] = a_after_tracklet_idx
                after_bad_scores[i] = b_after_score
                after_bad_choices[i] = b_after_tracklet_idx
            else:
                after_scores[i] = b_after_score
                after_choices[i] = b_after_tracklet_idx
                after_bad_scores[i] = a_after_score
                after_bad_choices[i] = a_after_tracklet_idx

        all_scores = before_scores + before_bad_scores + after_scores + after_bad_scores
        all_choices = (
            before_choices + before_bad_choices + after_choices + after_bad_choices
        )
        connections = [
            [None, None] for _ in range(len(common_tracklets))
        ]  # (before, after) forward
        tracklets_frames = [set(x[5]) for x in common_tracklets]
        for x in np.argsort(all_scores)[::-1]:
            cur_choice = all_choices[x]
            if cur_choice == "placeholder":
                if x < 2 * len(common_tracklets):  # before
                    x %= len(common_tracklets)
                    if connections[x][0] is None:
                        connections[x][0] = "placeholder"
                else:  # after
                    x %= len(common_tracklets)
                    if connections[x][1] is None:
                        connections[x][1] = "placeholder"
                continue
            if x < 2 * len(common_tracklets):  # before
                x %= len(common_tracklets)
                if (
                    connections[x][0] is None
                    and connections[cur_choice][1] is None
                    and tracklets_frames[cur_choice].isdisjoint(tracklets_frames[x])
                ):
                    connections[x][0] = cur_choice
                    connections[cur_choice][1] = x
                    union = tracklets_frames[x].union(
                        tracklets_frames[cur_choice])
                    tracklets_frames[x] = union
                    tracklets_frames[cur_choice] = union
            else:  # after
                x %= len(common_tracklets)
                if (
                    connections[x][1] is None
                    and connections[cur_choice][0] is None
                    and tracklets_frames[cur_choice].isdisjoint(tracklets_frames[x])
                ):
                    connections[x][1] = cur_choice
                    connections[cur_choice][0] = x
                    union = tracklets_frames[x].union(
                        tracklets_frames[cur_choice])
                    tracklets_frames[x] = union
                    tracklets_frames[cur_choice] = union

        merged_tracklet_inds_list = []
        visited = [False] * len(connections)
        for i, connection in enumerate(connections):
            if visited[i]:
                continue
            if connection[0] == "placeholder" or connection[0] is None:
                cur_tracklet = [i]
                visited[i] = True
                next_i = connection[1]
                while next_i != "placeholder" and next_i is not None:
                    cur_tracklet.append(next_i)
                    visited[next_i] = True
                    next_i = connections[next_i][1]
                merged_tracklet_inds_list.append(cur_tracklet)

        cur_merged_trajectories = {}
        for merged_tracklet_inds in merged_tracklet_inds_list:
            all_boxes = []
            all_objs = []

            for idx in merged_tracklet_inds:
                (
                    a_idx,
                    b_idx,
                    a_last_idx,
                    b_last_idx,
                    track_len,
                    cur_frames,
                ) = common_tracklets[idx]
                a_boxes, a_objs = a_trajectories[a_ids[a_idx]]
                b_boxes, b_objs = b_trajectories[b_ids[b_idx]]
                a_first_idx = a_last_idx - track_len + 1
                b_first_idx = len(b_boxes) - b_last_idx - 1  # reverse back
                b_last_idx = b_first_idx + track_len - 1

                # a, b are the same
                all_boxes += a_boxes[a_first_idx: a_last_idx + 1]
                all_objs += a_objs[a_first_idx: a_last_idx + 1]

            for x in all_objs:
                x.tracking_id = cur_id
            cur_merged_trajectories[cur_id] = (all_boxes, all_objs)
            cur_id += 1

        if visualize_contradictions:
            visualize_trajectories([x[0]
                                   for x in cur_merged_trajectories.values()])

        merged_trajectories.update(cur_merged_trajectories)

    return merged_trajectories
