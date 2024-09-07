import jieba
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

# Expanded training data with more diverse examples
sample_data = [
    ("糖尿病管理策略", "Relevant"),
    ("老年人高血压", "Relevant"),
    ("胆固醇治疗的最新进展", "Relevant"),
    ("办公室管理技巧", "Not Relevant"),
    ("医疗专业人员的压力管理", "Not Relevant"),
    ("血糖监测技术", "Relevant"),
    ("医疗保健中的有效沟通", "Not Relevant"),
    ("脂质谱分析和解释", "Relevant"),
    ("医疗设施设计原则", "Not Relevant"),
    ("胰岛素治疗进展", "Relevant"),
    ("糖尿病并发症预防", "Relevant"),
    ("高血压与心血管疾病", "Relevant"),
    ("糖尿病患者的饮食管理", "Relevant"),
    ("医疗保险政策解读", "Not Relevant"),
    ("慢性肾病与糖尿病", "Relevant"),
    ("医疗器械维护", "Not Relevant"),
    ("糖尿病足部护理", "Relevant"),
    ("医院感染控制", "Not Relevant"),
    ("糖尿病与妊娠", "Relevant"),
    ("医疗文书写作", "Not Relevant"),
    ("高血压与肾脏疾病", "Relevant"),
    ("代谢综合征的诊断与治疗", "Relevant"),
    ("心血管疾病风险评估", "Relevant"),
    ("糖尿病肾病的早期诊断", "Relevant"),
    ("胰岛素泵治疗技术", "Relevant"),
    ("糖尿病视网膜病变筛查", "Relevant"),
    ("医疗保健系统管理", "Not Relevant"),
    ("医院财务管理", "Not Relevant"),
    ("医疗设备采购流程", "Not Relevant"),
    ("医院信息系统设计", "Not Relevant"),
    ("中医治疗糖尿病", "Relevant"),
    ("针灸对高血压的影响", "Relevant"),
    ("中西医结合治疗代谢综合征", "Relevant"),
    ("医院人力资源管理", "Not Relevant"),
    ("糖尿病患者的运动处方", "Relevant"),
    ("高血压患者的饮食指导", "Relevant"),
    ("糖尿病与认知功能障碍", "Relevant"),
    ("医疗数据分析与决策支持", "Not Relevant"),
    ("糖尿病药物相互作用", "Relevant"),
    ("高血压与睡眠障碍", "Relevant")
]

def train_and_evaluate_model(data):
    try:
        df = pd.DataFrame(data, columns=['text', 'label'])
        
        # Preprocess the text data
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        X = df['processed_text']
        y = df['label']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create pipelines for SVM and Random Forest
        svm_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 3))),
            ('scaler', StandardScaler(with_mean=False)),
            ('svm', SVC(probability=True, random_state=42))
        ])
        
        rf_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 3))),
            ('rf', RandomForestClassifier(random_state=42))
        ])
        
        # Define parameter grids for GridSearchCV
        svm_param_grid = {
            'svm__C': [0.1, 1, 10],
            'svm__kernel': ['linear', 'rbf']
        }
        
        rf_param_grid = {
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [None, 10, 20]
        }
        
        # Perform GridSearchCV for both models
        svm_grid = GridSearchCV(svm_pipeline, svm_param_grid, cv=5, scoring='f1')
        rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='f1')
        
        svm_grid.fit(X_train, y_train)
        rf_grid.fit(X_train, y_train)
        
        # Select the best model
        if svm_grid.best_score_ > rf_grid.best_score_:
            best_model = svm_grid.best_estimator_
            logger.info("SVM model selected")
        else:
            best_model = rf_grid.best_estimator_
            logger.info("Random Forest model selected")
        
        # Evaluate the best model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        logger.info(f"Model performance metrics:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        return best_model, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    except Exception as e:
        logger.error(f"Error in train_and_evaluate_model: {str(e)}")
        return None, None

def preprocess_text(text):
    try:
        # Use jieba for Chinese text segmentation
        tokens = jieba.cut(text)
        # Join the tokens back into a string, as TfidfVectorizer will handle the tokenization
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error in preprocessing text: {str(e)}")
        return text

def perform_ai_review(table_data):
    try:
        # Prepare data for model training
        review_data = [(f"{row['Topic']} {row['Topic Description']}", "Relevant") for row in table_data]
        review_data.extend(sample_data)  # Add sample data to improve model performance
        
        # Train and evaluate the model
        model, metrics = train_and_evaluate_model(review_data)
        
        if not model:
            logger.error("Failed to train the model.")
            return table_data
        
        for row in table_data:
            topic = row['Topic']
            description = row['Topic Description']
            
            try:
                # Preprocess and make prediction
                input_text = preprocess_text(f"{topic} {description}")
                prediction = model.predict([input_text])[0]
                confidence = np.max(model.predict_proba([input_text]))
                
                row['AI Preliminary Review'] = prediction
                row['AI Review Confidence'] = f"{confidence:.2%}"
                
                if prediction == 'Not Relevant':
                    row['AI Preliminary Review Credit Points'] = 0
                else:
                    row['AI Preliminary Review Credit Points'] = row['Credit Points']
            except Exception as e:
                logger.error(f"Error in AI review for topic '{topic}': {str(e)}")
                row['AI Preliminary Review'] = 'Error'
                row['AI Review Confidence'] = 'N/A'
                row['AI Preliminary Review Credit Points'] = 0
        
        generate_model_report(metrics)
        return table_data
    except Exception as e:
        logger.error(f"Error in perform_ai_review: {str(e)}")
        return table_data

def generate_model_report(metrics):
    report = f"""
    AI Preliminary Review Model Performance Report
    
    Accuracy: {metrics['accuracy']:.2%}
    Precision: {metrics['precision']:.2%}
    Recall: {metrics['recall']:.2%}
    F1 Score: {metrics['f1']:.2%}
    
    Observations:
    1. The model shows {metrics['accuracy']:.2%} accuracy in classifying topics as Relevant or Not Relevant.
    2. Precision of {metrics['precision']:.2%} indicates the model's ability to avoid false positives.
    3. Recall of {metrics['recall']:.2%} shows the model's capability to identify all relevant topics.
    4. F1 Score of {metrics['f1']:.2%} provides a balanced measure of the model's performance.
    
    Potential Improvements:
    1. Further increase the training data size with more diverse examples.
    2. Experiment with different feature extraction techniques or word embeddings.
    3. Consider using ensemble methods or deep learning models for potentially better performance on larger datasets.
    4. Implement regular model retraining to adapt to new patterns in the data.
    """
    
    logger.info("AI Model Performance Report:")
    logger.info(report)

# Initialize jieba
jieba.initialize()
