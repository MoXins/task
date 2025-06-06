import os
import json
import numpy as np
import cv2
import csv
import glob # 新增导入glob模块
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
import matplotlib
matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义数据路径
IMAGE_DIR = '/Users/lx/Documents/机器学习课设/mchar_train'
JSON_PATH = '/Users/lx/Documents/机器学习课设/tran.json'

# 加载JSON标注数据
def load_annotations(json_path):
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    return annotations

# 提取单个字符的图像和标签
def extract_characters(image_dir, annotations):
    X = []
    y = []
    
    total_images = len(annotations)
    processed = 0
    
    print(f"开始处理 {total_images} 张图片...")
    
    for img_name, anno in annotations.items():
        # 显示进度
        processed += 1
        if processed % 100 == 0 or processed == total_images:
            print(f"处理进度: {processed}/{total_images} ({processed/total_images*100:.1f}%)")
        
        # 读取图像
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue
            
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 获取每个字符的位置和标签
        heights = anno.get('height', [])
        lefts = anno.get('left', [])
        tops = anno.get('top', [])
        widths = anno.get('width', [])
        labels = anno.get('label', [])
        
        # 确保所有列表长度一致
        min_len = min(len(heights), len(lefts), len(tops), len(widths), len(labels))
        
        for i in range(min_len):
            # 提取字符区域
            x = lefts[i]
            y_coord = tops[i]
            w = widths[i]
            h = heights[i]
            
            # 确保坐标在图像范围内
            if x < 0 or y_coord < 0 or x + w > image.shape[1] or y_coord + h > image.shape[0]:
                continue
                
            char_img = gray[y_coord:y_coord+h, x:x+w]
            
            # 调整大小为固定尺寸 (32x32)
            char_img = cv2.resize(char_img, (32, 32))
            
            # 提取HOG特征
            hog_features = extract_hog_features(char_img)
            
            X.append(hog_features)
            y.append(labels[i])
            
            # 定期释放内存
            if len(X) % 1000 == 0:
                # 强制垃圾回收
                import gc
                gc.collect()
    
    print("特征提取完成！")
    return np.array(X), np.array(y)

# 使用HOG提取特征
def extract_hog_features(image):
    # HOG参数 - 减少特征维度以加快计算速度
    win_size = (32, 32)
    block_size = (16, 16)  # 增大block_size减少特征数量
    block_stride = (8, 8)  # 增大stride减少特征数量
    cell_size = (8, 8)     # 增大cell_size减少特征数量
    nbins = 9
    
    # 创建HOG描述符
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # 计算HOG特征
    hog_features = hog.compute(image)
    
    return hog_features.flatten()

# 训练模型并评估
def train_and_evaluate():
    print("加载标注数据...")
    annotations = load_annotations(JSON_PATH)
    
    print(f"共有 {len(annotations)} 张图片的标注")
    
    # 显示内存使用情况
    try:
        import psutil
        process = psutil.Process(os.getpid())
        print(f"当前内存使用: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    except ImportError:
        print("未安装psutil模块，无法监控内存使用")
    
    print("提取字符特征...")
    X, y = extract_characters(IMAGE_DIR, annotations)
    
    if len(X) == 0:
        print("没有提取到有效的字符数据！")
        return
        
    print(f"提取了 {len(X)} 个字符样本")
    
    # 再次显示内存使用情况
    try:
        import psutil
        process = psutil.Process(os.getpid())
        print(f"特征提取后内存使用: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    except ImportError:
        pass
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
    
    # 创建模型
    models = {
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', C=10, gamma='scale', probability=False, cache_size=500))
        ]),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }
    
    # 如果数据量过大，考虑减少特征维度
    if len(X_train) > 5000:
        print("数据量较大，考虑使用PCA降维...")
        try:
            from sklearn.decomposition import PCA
            # 添加PCA降维到SVM管道中
            models['SVM'] = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=min(100, X_train.shape[1]), random_state=42)),
                ('classifier', SVC(kernel='rbf', C=10, gamma='scale', verbose=True, max_iter=1000, tol=1e-3, cache_size=500))
            ])
            print("已添加PCA降维到SVM模型")
        except Exception as e:
            print(f"PCA降维设置失败: {e}")
            pass
    
    results = {}
    
    # 训练和评估每个模型
    for name, model in models.items():
        print(f"\n训练 {name} 模型...")
        
        # 对于SVM模型，使用更高效的设置和进度显示
        if name == 'SVM':
            print("设置SVM参数以提高训练效率...")
            # 修改SVM参数，提高训练速度
            model.set_params(classifier__verbose=True, classifier__max_iter=1000, classifier__tol=1e-3)
            # 如果数据量很大，可以考虑减少训练样本
            if len(X_train) > 10000:
                print(f"训练样本较多({len(X_train)}个)，随机抽取10000个进行训练...")
                indices = np.random.choice(len(X_train), min(10000, len(X_train)), replace=False)
                X_train_sample = X_train[indices]
                y_train_sample = y_train[indices]
                try:
                    # 设置超时机制
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("SVM训练超时，尝试使用线性核或其他分类器")
                    
                    # 设置60秒超时
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(60)
                    
                    # 尝试训练
                    model.fit(X_train_sample, y_train_sample)
                    
                    # 取消超时
                    signal.alarm(0)
                except TimeoutError as e:
                    print(f"警告: {e}")
                    print("切换到线性核SVM...")
                    # 切换到线性核，速度更快
                    model.set_params(classifier__kernel='linear')
                    model.fit(X_train_sample, y_train_sample)
                except Exception as e:
                    print(f"SVM训练出错: {e}")
                    print("尝试使用线性核SVM...")
                    model.set_params(classifier__kernel='linear')
                    model.fit(X_train_sample, y_train_sample)
            else:
                try:
                    # 设置超时机制
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("SVM训练超时，尝试使用线性核或其他分类器")
                    
                    # 设置60秒超时
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(60)
                    
                    # 尝试训练
                    model.fit(X_train, y_train)
                    
                    # 取消超时
                    signal.alarm(0)
                except TimeoutError as e:
                    print(f"警告: {e}")
                    print("切换到线性核SVM...")
                    # 切换到线性核，速度更快
                    model.set_params(classifier__kernel='linear')
                    model.fit(X_train, y_train)
                except Exception as e:
                    print(f"SVM训练出错: {e}")
                    print("尝试使用线性核SVM...")
                    model.set_params(classifier__kernel='linear')
                    model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        print(f"完成{name}模型训练，开始评估...")
        # 在测试集上评估
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{name} 准确率: {accuracy:.4f}")
        print("分类报告:")
        print(classification_report(y_test, y_pred))
        
        results[name] = accuracy
    
    # 找出最佳模型
    best_model_name = max(results, key=results.get)
    print(f"\n最佳模型是 {best_model_name}，准确率为 {results[best_model_name]:.4f}")
    
    # 保存最佳模型的混淆矩阵
    best_model = models[best_model_name]
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'混淆矩阵 - {best_model_name}')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 保存最佳模型
    print(f"保存最佳模型 {best_model_name} 到 best_model.pkl")
    import pickle
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    return best_model

# 预测新图像中的字符
def predict_image(model, image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return []
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 尝试进行简单的字符分割（基于连通区域分析）
    # 二值化图像
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 查找连通区域
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 按照x坐标排序连通区域（从左到右）
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    predictions = []
    
    # 如果找到连通区域，则分别处理每个区域
    if contours:
        for contour in contours:
            # 获取字符的边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 过滤掉太小的区域（可能是噪点）
            if w < 5 or h < 5:
                continue
                
            # 提取字符区域
            char_img = gray[y:y+h, x:x+w]
            
            # 调整大小为固定尺寸 (32x32)
            char_img = cv2.resize(char_img, (32, 32))
            
            # 提取HOG特征
            hog_features = extract_hog_features(char_img)
            
            # 预测
            prediction = model.predict([hog_features])[0]
            predictions.append(prediction)
    
    # 如果没有找到任何字符，则将整个图像作为一个字符处理
    if not predictions:
        char_img = cv2.resize(gray, (32, 32))
        hog_features = extract_hog_features(char_img)
        prediction = model.predict([hog_features])[0]
        predictions.append(prediction)
    
    return predictions

# 新增测评函数
def evaluate_submission(submission_csv_path, ground_truth_json_path):
    """根据提交的CSV文件和真实的JSON标注评估预测结果"""
    try:
        # 1. 加载真实标注数据
        print(f"加载真实标注数据从: {ground_truth_json_path}")
        with open(ground_truth_json_path, 'r') as f:
            ground_truth_annotations = json.load(f)
        
        true_codes = {}
        for img_name, anno in ground_truth_annotations.items():
            # 假设 'label' 键包含一个数字列表，代表编码的各个字符
            if 'label' in anno and isinstance(anno['label'], list):
                true_codes[img_name] = "".join(map(str, anno['label']))
            else:
                print(f"警告: 图像 {img_name} 的标注格式不正确或缺少 'label' 键，已跳过。")

        if not true_codes:
            print("错误: 未能从JSON文件中加载任何有效的真实编码数据。")
            return

        # 2. 读取提交的CSV文件
        print(f"读取提交的CSV文件: {submission_csv_path}")
        submitted_predictions = {}
        with open(submission_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'file_name' not in reader.fieldnames or 'file_code' not in reader.fieldnames:
                print("错误: CSV文件必须包含 'file_name' 和 'file_code' 列。")
                return
            for row in reader:
                submitted_predictions[row['file_name']] = str(row['file_code'])
        
        if not submitted_predictions:
            print("错误: 提交的CSV文件为空或格式不正确。")
            return

        # 3. 比较并计算得分
        correct_predictions = 0
        total_submitted_images = len(submitted_predictions)
        
        print(f"总共提交了 {total_submitted_images} 张图片的预测结果。")
        print(f"真实标注中包含 {len(true_codes)} 张图片的编码。")

        evaluated_count = 0
        for file_name, submitted_code in submitted_predictions.items():
            if file_name in true_codes:
                evaluated_count += 1
                if submitted_code == true_codes[file_name]:
                    correct_predictions += 1
                # else:
                #     print(f"图像 {file_name}: 提交 {submitted_code}, 真实 {true_codes[file_name]}") # 可选：打印不匹配项
            else:
                print(f"警告: 提交的图像 {file_name} 在真实标注中未找到，已跳过评估。")

        if evaluated_count == 0:
            print("错误: 提交的图像均未在真实标注数据中找到，无法进行评估。")
            score = 0.0
        else:
            score = correct_predictions / evaluated_count
            print(f"在 {evaluated_count} 张可评估的图片中，正确识别了 {correct_predictions} 张。")

        print(f"\n测评得分 (准确率): {score:.4f}")
        
    except FileNotFoundError:
        print(f"错误: 文件未找到。请检查路径: {submission_csv_path} 或 {ground_truth_json_path}")
    except json.JSONDecodeError:
        print(f"错误: JSON文件格式无效: {ground_truth_json_path}")
    except Exception as e:
        print(f"评估过程中发生错误: {e}")

# 主函数
def load_model(model_path='best_model.pkl'):
    """加载保存的模型"""
    import pickle
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"成功加载模型: {model_path}")
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='字符识别模型训练与预测')
    parser.add_argument('--train', action='store_true', help='训练新模型')
    parser.add_argument('--predict', type=str, help='预测图像路径')
    parser.add_argument('--evaluate_submission', type=str, help='评估提交的CSV文件路径，需要提供CSV文件路径') # <-- 新增参数
    args = parser.parse_args()
    
    if args.evaluate_submission:
        print(f"开始评估提交结果: {args.evaluate_submission}")
        # 假设真实标注文件路径是全局定义的 JSON_PATH
        evaluate_submission(args.evaluate_submission, JSON_PATH)
    elif args.train:
        print("开始训练字符识别模型...")
        best_model = train_and_evaluate()
    else:
        # 尝试加载已有模型
        best_model = load_model()
    
    if best_model is not None and args.predict:
        # 预测指定图像
        predictions = predict_image(best_model, args.predict)
        print(f"预测结果: {''.join(map(str, predictions))}")
    elif best_model is not None:
        # 示例：预测一张图像
        test_image = os.path.join(IMAGE_DIR, '000000.png')  # 使用第一张图像作为测试
        predictions = predict_image(best_model, test_image)
        print(f"预测结果: {''.join(map(str, predictions))}")
        
        # 显示如何使用该模型进行预测的说明
        print("\n使用方法:")
        print("1. 训练新模型: python study.py --train")
        print("2. 预测图像: python study.py --predict 图像路径")

def batch_predict(model, image_dir, output_csv):
    """
    批量预测函数
    :param model: 训练好的模型
    :param image_dir: 测试图片目录
    :param output_csv: 输出CSV文件路径
    """
    # 获取所有测试图片
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg')) + \
                 glob.glob(os.path.join(image_dir, '*.png')) + \
                 glob.glob(os.path.join(image_dir, '*.jpeg'))
    
    # 创建CSV文件并写入表头
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file', 'prediction'])
        
        # 处理每张图片并显示进度
        from tqdm import tqdm
        for img_path in tqdm(image_paths, desc='预测进度'):
            # 读取并预处理图像
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 尝试进行字符分割
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 如果没有找到字符区域，则跳过
            if not contours:
                continue
                
            # 按照x坐标排序连通区域（从左到右）
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
            
            # 处理每个字符区域
            predictions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                char_img = gray[y:y+h, x:x+w]
                char_img = cv2.resize(char_img, (32, 32))
                
                # 提取HOG特征
                features = extract_hog_features(char_img)
                
                # 预测单个字符
                prediction = model.predict([features])[0]
                predictions.append(str(prediction))
            
            # 合并预测结果
            prediction = ''.join(predictions)
            
            # 写入CSV
            writer.writerow([os.path.basename(img_path), prediction])
    
    print(f"预测结果已保存到 {output_csv}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='字符识别模型训练和预测')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--predict', type=str, help='预测单个图像路径')
    parser.add_argument('--batch_predict', type=str, help='批量预测图片目录')
    parser.add_argument('--output_csv', type=str, default='predictions.csv', 
                       help='批量预测结果输出CSV文件路径')
    
    args = parser.parse_args()
    
    if args.train:
        train_and_evaluate()
    elif args.predict:
        # 单个预测逻辑
        pass
    elif args.batch_predict:
        # 加载最佳模型
        try:
            import joblib
            model = joblib.load('best_model.pkl')
            batch_predict(model, args.batch_predict, args.output_csv)
        except Exception as e:
            print(f"加载模型失败: {e}")
    else:
        print("请指定--train或--predict参数")