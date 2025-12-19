import os
import datetime
import cv2
import numpy as np
import logging
from src.recognition import SymbolRecognizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_dir):
    """
    加载训练数据
    """
    images = []
    labels = []
    
    if not os.path.exists(data_dir):
        logger.error(f"数据目录不存在: {data_dir}")
        return images, labels
        
    logger.info(f"正在从 {data_dir} 加载数据...")
    
    # 遍历所有子目录（类别）
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    total_files = 0
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        for filename in files:
            filepath = os.path.join(category_dir, filename)
            
            # 读取图像（灰度模式）
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                images.append(img)
                labels.append(category)
                total_files += 1
            else:
                logger.warning(f"无法读取图像: {filepath}")
                
        if len(images) % 1000 == 0 and len(images) > 0:
            logger.info(f"已加载 {len(images)} 张图像...")
            
    logger.info(f"数据加载完成，共 {len(images)} 张图像，{len(categories)} 个类别")
    return images, labels

def main():
    # 数据目录
    data_dir = os.path.join('data', 'training')
    
    # 生成带时间戳的模型文件名
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    model_filename = f"model_{timestamp}.pkl"
    
    # 模型保存路径
    model_path = os.path.join('models', model_filename)
    
    # 1. 加载数据
    images, labels = load_data(data_dir)
    
    if not images:
        logger.error("未加载到任何数据，终止训练")
        return
        
    # 2. 初始化识别器
    logger.info("初始化识别器...")
    recognizer = SymbolRecognizer()
    
    # 3. 训练模型
    # 使用 SVM 分类器，也可以尝试 'rf' (随机森林) 或 'knn'
    logger.info("开始训练模型...")
    try:
        report = recognizer.train(images, labels, classifier_type='svm')
        
        # 打印分类报告摘要
        logger.info("训练报告摘要:")
        logger.info(f"准确率: {report['accuracy']:.4f}")
        logger.info(f"宏平均 F1: {report['macro avg']['f1-score']:.4f}")
        logger.info(f"加权平均 F1: {report['weighted avg']['f1-score']:.4f}")
        
        # 4. 保存模型
        recognizer.save_model(model_path)
        logger.info("训练流程完成！")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)

if __name__ == '__main__':
    main()
