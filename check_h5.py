import h5py
from pathlib import Path

def inspect_segment_item():
    h5_path = Path("data/processed/head_pose_full_v3.h5")
    
    with h5py.File(h5_path, "r") as f:
        user_group = f['users']
        for user_id in user_group.keys():
            user_sub = user_group[user_id]
            if 'sessions' in user_sub:
                sessions_group = user_sub['sessions']
                for session_name in sessions_group.keys():
                    session_group = sessions_group[session_name]
                    if 'segments' in session_group:
                        seg = session_group['segments']
                        if '0000' in seg:
                            item = seg['0000']
                            print(f"📁 Kiểu của '0000': {type(item)}")
                            
                            if isinstance(item, h5py.Group):
                                print(f"   └─ Các thành phần con trong '0000': {list(item.keys())}")
                                for k in item.keys():
                                    sub = item[k]
                                    if hasattr(sub, 'shape'):
                                        print(f"      • [{k}] shape: {sub.shape}, dtype: {sub.dtype}")
                            elif hasattr(item, 'shape'):
                                print(f"   └─ Shape của '0000': {item.shape}, dtype: {item.dtype}")
                    return  # Dừng ở mẫu đầu tiên

if __name__ == "__main__":
    inspect_segment_item()