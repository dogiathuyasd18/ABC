import h5py
import pandas as pd
from pathlib import Path

def analyze_dataset_stats():
    h5_path = Path("data/processed/head_pose_v2.h5")
    filter_csv = Path("data/processed/session_filter.csv")
    reject_csv = Path("data/processed/session_rejects.csv")
    passed_csv = Path("data/processed/passed_sessions.csv")
    
    print("=" * 70)
    print("📊 BÁO CÁO THỐNG KÊ & ĐÁNH GIÁ CHẤT LƯỢNG DATASET HDF5")
    print("=" * 70)
    
    # 1. Thống kê trực tiếp từ file HDF5 v3
    if not h5_path.exists():
        print(f"❌ Không tìm thấy file HDF5 tại: {h5_path}")
        return
    
    total_users = 0
    total_sessions_h5 = 0
    total_segments = 0
    total_frames = 0
    
    with h5py.File(h5_path, "r") as f:
        if 'users' in f:
            users_grp = f['users']
            total_users = len(users_grp.keys())
            for user_id in users_grp.keys():
                user_sub = users_grp[user_id]
                if 'sessions' in user_sub:
                    sess_grp = user_sub['sessions']
                    total_sessions_h5 += len(sess_grp.keys())
                    for sess_id in sess_grp.keys():
                        sess_item = sess_grp[sess_id]
                        if 'segments' in sess_item:
                            segs = sess_item['segments']
                            total_segments += len(segs.keys())
                            for seg_id in segs.keys():
                                if 'quaternions' in segs[seg_id]:
                                    total_frames += segs[seg_id]['quaternions'].shape[0]

    print(f"👤 Tổng số User có trong HDF5: {total_users}")
    print(f"📁 Tổng số Session hợp lệ được lưu: {total_sessions_h5}")
    print(f"🧩 Tổng số Segment: {total_segments}")
    print(f"🎞️ Tổng số lượng Frame dữ liệu (quaternions): {total_frames:,}")
    print("-" * 70)
    
    # 2. Thống kê qua các file CSV Log (nếu có)
    print("🔍 PHÂN TÍCH TỪ NHẬT KÝ LỌC DỮ LIỆU (CSV LOGS):")
    
    if filter_csv.exists():
        df_filter = pd.read_csv(filter_csv)
        print(f"📄 {filter_csv.name}: Ghi nhận tổng cộng {len(df_filter)} bản ghi xử lý.")
    else:
        print(f"⚠️ Không tìm thấy file: {filter_csv.name}")

    if passed_csv.exists():
        df_passed = pd.read_csv(passed_csv)
        print(f"✅ Số lượng session ĐẠT CHUẨN (Passed): {len(df_passed)}")
    
    if reject_csv.exists():
        df_rejects = pd.read_csv(reject_csv)
        print(f"❌ Số lượng session BỊ LOẠI (Rejected): {len(df_rejects)}")
        if 'reason' in df_rejects.columns:
            print("\n   📌 Top các lý do bị từ chối phổ biến:")
            reason_counts = df_rejects['reason'].value_counts().head(5)
            for reason, count in reason_counts.items():
                print(f"      - {reason}: {count} lần")
    else:
        print(f"ℹ️ Không có file session_rejects.csv hoặc chưa có session nào bị loại.")
        
    print("=" * 70)
    print("💡 ĐÁNH GIÁ ĐỘ TIN CẬY ĐỂ HUẤN LUYỆN (RELIABILITY CHECK):")
    
    # Tiêu chuẩn đánh giá nhanh
    is_reliable = True
    messages = []
    
    if total_sessions_h5 < 20:
        is_reliable = False
        messages.append("⚠️ Số lượng session hơi ít (< 20), mô hình dễ bị overfit.")
    else:
        messages.append(f"✔️ Số lượng session ({total_sessions_h5}) đủ lớn để huấn luyện cơ bản.")
        
    if total_frames < 50000:
        messages.append("⚠️ Tổng số lượng frame thấp hơn 50,000, có thể thiếu đa dạng chuyển động.")
    else:
        messages.append(f"✔️ Tổng số frame lớn ({total_frames:,}), đảm bảo dữ liệu phong phú cho cửa sổ trượt (sliding windows).")
        
    if total_users <= 1:
        messages.append("⚠️ Dataset chỉ có 1 user duy nhất! Việc chia tập train/val dựa trên user split có thể gặp khó khăn hoặc không phản ánh đúng khả năng tổng quát hóa trên người dùng mới.")
    else:
        messages.append(f"✔️ Có {total_users} user khác nhau, rất lý tưởng để thực hiện chia tách tập train/val theo user (tránh data leakage).")

    for msg in messages:
        print(f"  • {msg}")
        
    print("=" * 70)

if __name__ == "__main__":
    analyze_dataset_stats()