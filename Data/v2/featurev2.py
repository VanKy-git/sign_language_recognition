import numpy as np

def interpolate_missing_landmarks(sequence):
    seq = np.array(sequence)
    mask = np.sum(np.abs(seq), axis=1) == 0
    valid_indices = np.where(~mask)[0]
    
    if len(valid_indices) == 0: return seq
        
    for i in range(len(seq)):
        if mask[i]:
            prev_idx = valid_indices[valid_indices < i]
            next_idx = valid_indices[valid_indices > i]
            if len(prev_idx) > 0: seq[i] = seq[prev_idx[-1]]
            elif len(next_idx) > 0: seq[i] = seq[next_idx[0]]
    return seq

def calculate_relative_coordinates(pose, left_hand, right_hand, face):
    left_shoulder = pose[0:3]
    right_shoulder = pose[3:6]
    neck = (left_shoulder + right_shoulder) / 2.0
    
    rel_pose = pose - np.tile(neck, len(pose)//3)
    rel_lh = left_hand - np.tile(neck, len(left_hand)//3) if np.sum(np.abs(left_hand)) > 0 else left_hand
    rel_rh = right_hand - np.tile(neck, len(right_hand)//3) if np.sum(np.abs(right_hand)) > 0 else right_hand
    rel_face = face - np.tile(neck, len(face)//3) if np.sum(np.abs(face)) > 0 else face
    
    return np.concatenate([rel_pose, rel_lh, rel_rh, rel_face])

def calculate_hand_angles(hand_landmarks):
    if np.sum(np.abs(hand_landmarks)) == 0: return np.zeros(15) 
    pts = hand_landmarks.reshape(21, 3)
    angles = []
    angle_indices = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),       
        (0, 5, 6), (5, 6, 7), (6, 7, 8),       
        (0, 9, 10), (9, 10, 11), (10, 11, 12), 
        (0, 13, 14), (13, 14, 15), (14, 15, 16),
        (0, 17, 18), (17, 18, 19), (18, 19, 20) 
    ]
    for (a, b, c) in angle_indices:
        v1, v2 = pts[a] - pts[b], pts[c] - pts[b]
        v1_norm, v2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
        if v1_norm == 0 or v2_norm == 0:
            angles.append(0.0)
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0) 
        angles.append(np.arccos(cos_angle))
    return np.array(angles)

def pad_or_sample_sequence(sequence, target_frames=50):
    length = len(sequence)
    if length == target_frames: return np.array(sequence)
    elif length > target_frames:
        indices = np.linspace(0, length - 1, target_frames).astype(int)
        return np.array(sequence)[indices]
    else:
        padding = np.tile(sequence[-1], (target_frames - length, 1))
        return np.vstack([sequence, padding])

def process_single_video_features(raw_frames):
    if len(raw_frames) == 0: return None
    frames_array = np.array(raw_frames)
    
    # Chia mảng 222 thô thành 4 khúc
    pose_part = frames_array[:, :36]
    lh_part = frames_array[:, 36:99]
    rh_part = frames_array[:, 99:162]
    face_part = frames_array[:, 162:222]
    
    lh_interp = interpolate_missing_landmarks(lh_part)
    rh_interp = interpolate_missing_landmarks(rh_part)
    
    rel_frames = []
    for i in range(len(frames_array)):
        # 1. Tọa độ tương đối (222 features)
        rel_coords = calculate_relative_coordinates(pose_part[i], lh_interp[i], rh_interp[i], face_part[i])
        # 2. Góc tay (30 features)
        lh_angles = calculate_hand_angles(lh_interp[i])
        rh_angles = calculate_hand_angles(rh_interp[i])
        
        # Ghép lại = 252 features
        rel_frames.append(np.concatenate([rel_coords, lh_angles, rh_angles]))
        
    fixed_frames = pad_or_sample_sequence(np.array(rel_frames), 50)
    deltas = np.zeros_like(fixed_frames)
    deltas[1:] = fixed_frames[1:] - fixed_frames[:-1]
    
    # Không gian (252) + Vận tốc (252) = 504 Dimensions
    return np.concatenate([fixed_frames, deltas], axis=1)