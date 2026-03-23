import numpy as np

def interpolate_missing_landmarks(sequence):
    seq = np.array(sequence)
    mask = np.sum(np.abs(seq), axis=1) == 0
    valid_indices = np.where(~mask)[0]
    
    if len(valid_indices) == 0:
        return seq
        
    for i in range(len(seq)):
        if mask[i]:
            prev_idx = valid_indices[valid_indices < i]
            next_idx = valid_indices[valid_indices > i]
            if len(prev_idx) > 0:
                seq[i] = seq[prev_idx[-1]]
            elif len(next_idx) > 0:
                seq[i] = seq[next_idx[0]]
    return seq

def calculate_relative_coordinates(pose, left_hand, right_hand):
    left_shoulder = pose[0:3]
    right_shoulder = pose[3:6]
    neck = (left_shoulder + right_shoulder) / 2.0
    
    rel_pose = pose - np.tile(neck, len(pose)//3)
    rel_lh = left_hand - np.tile(neck, len(left_hand)//3) if np.sum(np.abs(left_hand)) > 0 else left_hand
    rel_rh = right_hand - np.tile(neck, len(right_hand)//3) if np.sum(np.abs(right_hand)) > 0 else right_hand
    
    return np.concatenate([rel_pose, rel_lh, rel_rh])

# --- HÀM MỚI: TÍNH GÓC KHỚP TAY ---
def calculate_hand_angles(hand_landmarks):
    """Tính 15 góc bên trong của các đốt ngón tay."""
    if np.sum(np.abs(hand_landmarks)) == 0:
        return np.zeros(15) # Nếu mất dấu tay, góc = 0
        
    # Chuyển mảng 1D (63,) thành 2D (21 điểm, 3 trục xyz)
    pts = hand_landmarks.reshape(21, 3)
    angles = []
    
    # Bộ 3 điểm tạo thành góc: (A, B, C) -> Tính góc tại điểm B
    angle_indices = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),       # Ngón cái
        (0, 5, 6), (5, 6, 7), (6, 7, 8),       # Ngón trỏ
        (0, 9, 10), (9, 10, 11), (10, 11, 12), # Ngón giữa
        (0, 13, 14), (13, 14, 15), (14, 15, 16),# Ngón áp út
        (0, 17, 18), (17, 18, 19), (18, 19, 20) # Ngón út
    ]
    
    for (a, b, c) in angle_indices:
        v1 = pts[a] - pts[b]
        v2 = pts[c] - pts[b]
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            angles.append(0.0)
            continue
            
        dot_product = np.dot(v1, v2)
        cos_angle = dot_product / (v1_norm * v2_norm)
        # Ép giá trị vào khoảng [-1, 1] để tránh lỗi float của numpy sinh ra NaN
        cos_angle = np.clip(cos_angle, -1.0, 1.0) 
        angle = np.arccos(cos_angle)
        angles.append(angle)
        
    return np.array(angles)

def pad_or_sample_sequence(sequence, target_frames=50):
    length = len(sequence)
    if length == target_frames:
        return np.array(sequence)
    elif length > target_frames:
        indices = np.linspace(0, length - 1, target_frames).astype(int)
        return np.array(sequence)[indices]
    else:
        padding_length = target_frames - length
        last_frame = sequence[-1]
        padding = np.tile(last_frame, (padding_length, 1))
        return np.vstack([sequence, padding])

def process_single_video_features(raw_frames):
    if len(raw_frames) == 0: return None
    frames_array = np.array(raw_frames)
    
    pose_part = frames_array[:, :36]
    lh_part = frames_array[:, 36:99]
    rh_part = frames_array[:, 99:]
    
    lh_interp = interpolate_missing_landmarks(lh_part)
    rh_interp = interpolate_missing_landmarks(rh_part)
    
    rel_frames = []
    for i in range(len(frames_array)):
        p = pose_part[i]
        lh = lh_interp[i]
        rh = rh_interp[i]
        
        # 1. Tọa độ tương đối
        rel_coords = calculate_relative_coordinates(p, lh, rh)
        
        # 2. Tính góc tay (15 góc tay trái + 15 góc tay phải = 30 đặc trưng mới)
        lh_angles = calculate_hand_angles(lh)
        rh_angles = calculate_hand_angles(rh)
        
        # Ghép Tọa độ (162) + Góc (30) = 192 features
        combined_frame = np.concatenate([rel_coords, lh_angles, rh_angles])
        rel_frames.append(combined_frame)
        
    rel_frames = np.array(rel_frames)
    fixed_frames = pad_or_sample_sequence(rel_frames, 50)
    
    # Tính Vận tốc (Delta) trên cả tọa độ LẪN GÓC (Vận tốc góc)
    deltas = np.zeros_like(fixed_frames)
    deltas[1:] = fixed_frames[1:] - fixed_frames[:-1]
    
    # Nối lại: 192 (Không gian) + 192 (Thời gian) = 384 dimensions
    final_features = np.concatenate([fixed_frames, deltas], axis=1)
    return final_features