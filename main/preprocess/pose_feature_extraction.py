import numpy as np
import pandas as pd

pose_columns = ['spine_base.X', 'spine_base.Y', 'spine_base.Z', 'spine.X', 'spine.Y',
                'spine.Z', 'neck.X', 'neck.Y', 'neck.Z', 'spine_shoulder.X',
                'spine_shoulder.Y', 'spine_shoulder.Z', 'head.X', 'head.Y', 'head.Z',
                'left_shoulder.X', 'left_shoulder.Y', 'left_shoulder.Z', 'left_elbow.X',
                'left_elbow.Y', 'left_elbow.Z', 'left_wrist.X', 'left_wrist.Y',
                'left_wrist.Z', 'left_hand.X', 'left_hand.Y', 'left_hand.Z',
                'right_shoulder.X', 'right_shoulder.Y', 'right_shoulder.Z',
                'right_elbow.X', 'right_elbow.Y', 'right_elbow.Z', 'right_wrist.X',
                'right_wrist.Y', 'right_wrist.Z', 'right_hand.X', 'right_hand.Y',
                'right_hand.Z', 'left_hip.X', 'left_hip.Y', 'left_hip.Z', 'left_knee.X',
                'left_knee.Y', 'left_knee.Z', 'left_ankle.X', 'left_ankle.Y',
                'left_ankle.Z', 'left_foot.X', 'left_foot.Y', 'left_foot.Z',
                'right_hip.X', 'right_hip.Y', 'right_hip.Z', 'right_knee.X',
                'right_knee.Y', 'right_knee.Z', 'right_ankle.X', 'right_ankle.Y',
                'right_ankle.Z', 'right_foot.X', 'right_foot.Y', 'right_foot.Z',
                'left_fingertip.X', 'left_fingertip.Y', 'left_fingertip.Z',
                'left_thumb.X', 'left_thumb.Y', 'left_thumb.Z', 'right_fingertip.X',
                'right_fingertip.Y', 'right_fingertip.Z', 'right_thumb.X',
                'right_thumb.Y', 'right_thumb.Z']


def open_skeleton(filename: str, elements_per_sec=5) -> np.ndarray:
    df = pd.read_csv(filename, index_col=None, header=0)
    step = 1 / elements_per_sec
    if 'Time' in df.columns:
        df = df.sort_values('Time')
        max_ = df['Time'].max()
        grouped = df.groupby(pd.cut(df['Time'], np.arange(0, max_ + step, step))).mean()
        return grouped.drop(['Frame', 'Time'], axis=1).to_numpy()
    else:
        df = df.sort_values('Frame')
        max_ = df['Frame'].max()
        grouped = df.groupby(pd.cut(df['Frame'], np.arange(0, max_ + step, elements_per_sec))).mean()
        return grouped.drop(['Frame'], axis=1).to_numpy()


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return (180 / np.pi) * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def distance_between(v1, v2):
    return np.linalg.norm(np.asarray(v1) - np.asarray(v2))


def heron(a, b, c):
    s = (a + b + c) / 2
    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    return area


def area_triangle(v1, v2, v3):
    a = distance_between(v1, v2)
    b = distance_between(v1, v3)
    c = distance_between(v2, v3)
    A = heron(a, b, c)
    return A


# Volume of the bounding box
def compute_feature0_per_frame(frame):
    min_x = float('inf')
    min_y = float('inf')
    min_z = float('inf')

    max_x = float('-inf')
    max_y = float('-inf')
    max_z = float('-inf')
    for i in range(25):
        if min_x > frame[3 * i]:
            min_x = frame[3 * i]
        elif max_x < frame[3 * i]:
            max_x = frame[3 * i]

        if min_y > frame[3 * i + 1]:
            min_y = frame[3 * i + 1]
        elif max_y < frame[3 * i + 1]:
            max_y = frame[3 * i + 1]

        if min_z > frame[3 * i + 2]:
            min_z = frame[3 * i + 2]
        elif max_z < frame[3 * i + 2]:
            max_z = frame[3 * i + 2]
    volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
    volume = volume / 1000
    return volume


def compute_feature0(frames):
    array = []
    for frame in frames:
        if np.all(frame == 0.0):
            continue
        array.append(compute_feature0_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Angle at neck by shoulders
def compute_feature_1_per_frame(frame):
    jid = 9
    r_shoulder = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 2
    neck = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 5
    l_shoulder = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    angle = angle_between(r_shoulder - neck, l_shoulder - neck)
    return angle


def compute_feature_1(frames):
    array = []
    for frame in frames:
        if np.all(frame == 0.0):
            continue
        array.append(compute_feature_1_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Angle at right shoulder by neck and left shoulder
def compute_feature_2_per_frame(frame):
    jid = 9
    r_shoulder = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 2
    neck = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 5
    l_shoulder = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    angle = angle_between(neck - r_shoulder, l_shoulder - r_shoulder)
    return angle


def compute_feature_2(frames):
    array = []
    for frame in frames:
        if np.all(frame == 0.0):
            continue
        array.append(compute_feature_2_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Angle at left shoulder by neck and right shoulder
def compute_feature_3_per_frame(frame):
    jid = 9
    r_shoulder = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 2
    neck = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 5
    l_shoulder = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    angle = angle_between(neck - l_shoulder, r_shoulder - l_shoulder)
    return angle


def compute_feature_3(frames):
    array = []
    for frame in frames:
        if np.all(frame == 0.0):
            continue
        array.append(compute_feature_3_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Angle at neck by vertical and back
def compute_feature_4_per_frame(frame):
    jid = 4
    head = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 3
    root = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    up = np.asarray([0.0, 1.0, 0.0])
    angle = angle_between(head - root, up)
    return angle


def compute_feature_4(frames):
    array = []
    for frame in frames:
        if np.all(frame == 0.0):
            continue
        array.append(compute_feature_4_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Angle at neck by head and back
def compute_feature_5_per_frame(frame):
    jid = 4
    head = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 2
    neck = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 3
    spine = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    angle = angle_between(head - neck, spine - neck)
    return angle


def compute_feature_5(frames):
    array = []
    for frame in frames:
        if np.all(frame == 0.0):
            continue
        array.append(compute_feature_5_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Distance between right hand and root
def compute_feature_6_per_frame(frame):
    jid = 12
    hand = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 3
    root = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    distance = distance_between(hand, root)
    return distance / 10


def compute_feature_6(frames):
    array = []
    for frame in frames:
        if np.all(frame == 0.0):
            continue
        array.append(compute_feature_6_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Distance between left hand and root
def compute_feature_7_per_frame(frame):
    jid = 8
    hand = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 3
    root = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    distance = distance_between(hand, root)
    return distance / 10


def compute_feature_7(frames):
    array = []
    for frame in frames:
        if np.all(frame == 0.0):
            continue
        array.append(compute_feature_7_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Distance between right foot and root
def compute_feature_8_per_frame(frame):
    jid = 20
    foot = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 3
    root = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    distance = distance_between(foot, root)
    return distance / 10


def compute_feature_8(frames):
    array = []
    for frame in frames:
        if np.all(frame == 0.0):
            continue
        array.append(compute_feature_8_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Distance between left foot and root
def compute_feature_9_per_frame(frame):
    jid = 16
    foot = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 3
    root = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    distance = distance_between(foot, root)
    return distance / 10


def compute_feature_9(frames):
    array = []
    for frame in frames:
        if np.all(frame == 0.0):
            continue
        array.append(compute_feature_9_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Area of triangle between hands and neck
def compute_feature_10_per_frame(frame):
    jid = 8
    l_hand = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 2
    neck = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 12
    r_hand = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    area = area_triangle(l_hand, neck, r_hand)
    return area / 100


def compute_feature_10(frames):
    array = []
    for frame in frames:
        if np.all(frame == 0.0):
            continue
        array.append(compute_feature_10_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Area of triangle between feet and root
def compute_feature_11_per_frame(frame):
    jid = 16
    l_foot = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 3
    root = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 20
    r_foot = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    area = area_triangle(l_foot, root, r_foot)
    return area / 100


def compute_feature_11(frames):
    array = []
    for frame in frames:
        if np.all(frame == 0.0):
            continue
        array.append(compute_feature_11_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Calculate speed
def calculate_speed(frames, time_step, jid):
    array = []
    old_position = np.asarray([frames[0][3 * jid], frames[0][3 * jid + 1], frames[0][3 * jid + 2]])
    for i in range(1, len(frames)):
        new_position = np.asarray([frames[i][3 * jid], frames[i][3 * jid + 1], frames[i][3 * jid + 2]])
        distance = distance_between(new_position, old_position) / 10
        array.append(distance / time_step)
        old_position = new_position.copy()
    array = np.asarray(array)
    return np.mean(array)


# Speed of right hand
def compute_feature_12(frames, time_step):
    return calculate_speed(frames, time_step, 12)


# Speed of left hand
def compute_feature_13(frames, time_step):
    return calculate_speed(frames, time_step, 8)


# Speed of head
def compute_feature_14(frames, time_step):
    return calculate_speed(frames, time_step, 4)


# Speed of right foot
def compute_feature_15(frames, time_step):
    return calculate_speed(frames, time_step, 20)


# Speed of left foot
def compute_feature_16(frames, time_step):
    return calculate_speed(frames, time_step, 16)


# Calculate acceleration
def calculate_acceleration(frames, time_step, jid):
    array = []
    old_position = np.asarray([frames[0][3 * jid], frames[0][3 * jid + 1], frames[0][3 * jid + 2]])
    new_position = np.asarray([frames[1][3 * jid], frames[1][3 * jid + 1], frames[1][3 * jid + 2]])
    old_velocity = (old_position - old_position) / time_step
    old_position = new_position.copy()
    for i in range(2, len(frames)):
        new_position = np.asarray([frames[i][3 * jid], frames[i][3 * jid + 1], frames[i][3 * jid + 2]])
        new_velocity = (old_position - old_position) / time_step
        acceleration = (new_velocity - old_velocity) / time_step
        acceleration_mag = np.linalg.norm(acceleration) / 10
        old_position = new_position.copy()
        old_velocity = new_velocity.copy()
        array.append(acceleration_mag)
    array = np.asarray(array)
    return np.mean(array)


# Acceleration of right hand
def compute_feature_17(frames, time_step):
    return calculate_acceleration(frames, time_step, 12)


# Acceleration of left hand
def compute_feature_18(frames, time_step):
    return calculate_acceleration(frames, time_step, 8)


# Acceleration of head
def compute_feature_19(frames, time_step):
    return calculate_acceleration(frames, time_step, 4)


# Acceleration of right foot
def compute_feature_20(frames, time_step):
    return calculate_acceleration(frames, time_step, 20)


# Acceleration of left foot
def compute_feature_21(frames, time_step):
    return calculate_acceleration(frames, time_step, 16)


# Calculate movement jerk
def calculate_movement_jerk(frames, time_step, jid):
    array = []
    old_position = np.asarray([frames[0][3 * jid], frames[0][3 * jid + 1], frames[0][3 * jid + 2]])
    new_position = np.asarray([frames[1][3 * jid], frames[1][3 * jid + 1], frames[1][3 * jid + 2]])
    old_velocity = (new_position - old_position) / time_step
    old_position = new_position.copy()
    new_position = np.asarray([frames[2][3 * jid], frames[2][3 * jid + 1], frames[2][3 * jid + 2]])
    new_velocity = (new_position - old_position) / time_step
    old_acceleration = (new_velocity - old_velocity) / time_step
    old_velocity = new_velocity.copy()
    old_position = new_position.copy()
    for i in range(3, len(frames)):
        new_position = np.asarray([frames[i][3 * jid], frames[i][3 * jid + 1], frames[i][3 * jid + 2]])
        new_velocity = (new_position - old_position) / time_step
        new_acceleration = (new_velocity - old_velocity) / time_step
        jerk = (new_acceleration - old_acceleration) / time_step
        jerk_mag = np.linalg.norm(jerk) / 10
        old_position = new_position.copy()
        old_velocity = new_velocity.copy()
        old_acceleration = new_acceleration.copy()
        array.append(jerk_mag)
    array = np.asarray(array)
    return np.mean(array)


# Movement jerk of right hand
def compute_feature_22(frames, time_step):
    return calculate_movement_jerk(frames, time_step, 12)


# Movement jerk of left hand
def compute_feature_23(frames, time_step):
    return calculate_movement_jerk(frames, time_step, 8)


# Movement jerk of head
def compute_feature_24(frames, time_step):
    return calculate_movement_jerk(frames, time_step, 4)


# Movement jerk of right foot
def compute_feature_25(frames, time_step):
    return calculate_movement_jerk(frames, time_step, 20)


# Movement jerk of left foot
def compute_feature_26(frames, time_step):
    return calculate_movement_jerk(frames, time_step, 16)


# hands diff
def compute_feature_27(frames, time_step):
    array = []
    for frame in frames:
        if np.all(frame == 0.0):
            continue
        array.append(compute_feature_5_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# hands diff
def compute_feature_27_per_frame(frame):
    jid = 8
    r_hand = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 12
    l_hand = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    return np.linalg.norm(r_hand - l_hand)


# hands diff
def compute_feature_27(frames):
    array = []
    for frame in frames:
        if np.all(frame == 0.0):
            continue

        array.append(compute_feature_27_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


def compute_features(frames, time_step):
    if np.all(frames == 0.0):
        return np.zeros(28)

    features = [
        # Volume
        compute_feature0(frames),
        # Angles
        compute_feature_1(frames),
        compute_feature_2(frames),
        compute_feature_3(frames),
        compute_feature_4(frames),
        compute_feature_5(frames),
        # Distances
        compute_feature_6(frames),
        compute_feature_7(frames),
        compute_feature_8(frames),
        compute_feature_9(frames),
        # Areas
        compute_feature_10(frames),
        compute_feature_11(frames),
        # Speeds
        compute_feature_12(frames, time_step),
        compute_feature_13(frames, time_step),
        compute_feature_14(frames, time_step),
        compute_feature_15(frames, time_step),
        compute_feature_16(frames, time_step),
        # Accelerations
        compute_feature_17(frames, time_step),
        compute_feature_18(frames, time_step),
        compute_feature_19(frames, time_step),
        compute_feature_20(frames, time_step),
        compute_feature_21(frames, time_step),
        # Movement Jerk
        compute_feature_22(frames, time_step),
        compute_feature_23(frames, time_step),
        compute_feature_24(frames, time_step),
        compute_feature_25(frames, time_step),
        compute_feature_26(frames, time_step),
        # distance
        compute_feature_27(frames),

    ]
    return features
