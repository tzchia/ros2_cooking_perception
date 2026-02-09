import os
import re
import numpy as np

def convert_to_yolo_seg(root_path):
    # 遍歷 train 和 val 資料夾
    sub_folders = ['train', 'val']
    
    for sub in sub_folders:
        folder_path = os.path.join(root_path, sub)
        if not os.path.exists(folder_path):
            continue
            
        print(f"正在處理: {folder_path}")
        
        # 讀取資料夾中所有檔案
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 使用正則表達式提取 class_id 和 np.array 內的座標
                # 格式範例: 0 np.array([[0.49, 0.32], ...])
                matches = re.findall(r'(\d+)\s+np\.array\(\[\[(.*?)\]\]\)', content, re.DOTALL)
                
                yolo_lines = []
                for class_id, coords_str in matches:
                    # 清理字串並轉換為數值列表
                    # 先去掉換行與多餘空格，將 [[x, y], [x, y]] 轉為純數字序列
                    coords_clean = coords_str.replace('[', '').replace(']', '').replace('\n', '').split(',')
                    coords_flat = [c.strip() for c in coords_clean if c.strip()]
                    
                    # 組合成 YOLO 格式: class_id x1 y1 x2 y2 ...
                    line = f"{class_id} " + " ".join(coords_flat)
                    yolo_lines.append(line)
                
                # 寫回檔案 (或是另存新檔，這裡建議先覆蓋或存至 labels 資料夾)
                if yolo_lines:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("\n".join(yolo_lines))
        
    print("轉換完成！")

# 設定您的 labels 資料夾路徑
if __name__ == "__main__":
    # 請確保此路徑正確指向包含 train/val 的 labels 資料夾
    target_directory = "dataset_yolo/manual/labels" 
    convert_to_yolo_seg(target_directory)