from decision_tree_model import HeartDiseaseDecisionTree
import os


def main():
    """Hàm chính để huấn luyện và đánh giá model Decision Tree"""
    
    # Tạo thư mục output
    output_dir = 'decision_tree_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("="*60)
    print("HUẤN LUYỆN MODEL DECISION TREE CHO DỰ ĐOÁN BỆNH TIM")
    print("="*60)
    
    # Khởi tạo model
    dt = HeartDiseaseDecisionTree(max_depth=None, criterion='gini', random_state=42)
    
    # Tải dữ liệu
    data_file = 'Heart_disease_cleveland_new(2).csv'
    dt.load_data(data_file)
    
    # Tiền xử lý (85% train, 15% test)
    dt.preprocess_data(test_size=0.15, random_state=42)
    
    # Tìm max_depth tốt nhất
    best_depth = dt.tune_max_depth(depth_range=range(1, 21), 
                                    save_path=os.path.join(output_dir, 'dt_depth_tuning.png'))
    
    # Cập nhật model với depth tốt nhất
    print(f"\nCập nhật model với max_depth = {best_depth}")
    dt.max_depth = best_depth
    dt.model = dt.model.__class__(
        max_depth=best_depth,
        criterion=dt.criterion,
        random_state=dt.random_state
    )
    
    # Huấn luyện model
    dt.train()
    
    # Đánh giá model
    results = dt.evaluate()
    
    # Vẽ các biểu đồ
    dt.plot_confusion_matrix(save_path=os.path.join(output_dir, 'dt_confusion_matrix.png'))
    dt.plot_roc_curve(save_path=os.path.join(output_dir, 'dt_roc_curve.png'))
    dt.plot_feature_importance(save_path=os.path.join(output_dir, 'dt_feature_importance.png'))
    dt.plot_tree_visualization(save_path=os.path.join(output_dir, 'dt_tree_visualization.png'))
    
    # Tùy chọn: Thực hiện GridSearch để tìm tham số tốt nhất
    use_gridsearch = True
    if use_gridsearch:
        print("\n" + "="*60)
        print("GRIDSEARCH ĐỂ TÌM THAM SỐ TỐI ƯU")
        print("="*60)
        
        best_params = dt.grid_search()
        
        # Đánh giá lại với tham số tốt nhất
        print("\nĐánh giá lại model với tham số tốt nhất...")
        results = dt.evaluate()
        dt.plot_confusion_matrix(save_path=os.path.join(output_dir, 'dt_confusion_matrix_optimized.png'))
        dt.plot_roc_curve(save_path=os.path.join(output_dir, 'dt_roc_curve_optimized.png'))
        dt.plot_feature_importance(save_path=os.path.join(output_dir, 'dt_feature_importance_optimized.png'))
        dt.plot_tree_visualization(save_path=os.path.join(output_dir, 'dt_tree_visualization_optimized.png'))
    
    # Lưu model sau khi train
    dt.save_model(
        model_path=os.path.join(output_dir, 'dt_heart_disease_model.pkl'),
        scaler_path=os.path.join(output_dir, 'dt_scaler.pkl')
    )
    
    # Xuất metrics ra Excel
    dt.export_metrics_to_excel(results, filename=os.path.join(output_dir, 'dt_metrics.xlsx'))
    
    print("\n" + "="*60)
    print("HOÀN THÀNH!")
    print("="*60)
    print("\nKết quả:")
    print(f"- Độ chính xác trên test set: {results['test_accuracy']*100:.2f}%")
    print(f"- Precision: {results['precision']:.4f}")
    print(f"- Recall: {results['recall']:.4f}")
    print(f"- F1-Score: {results['f1_score']:.4f}")
    print(f"- ROC AUC Score: {results['roc_auc']:.4f}")
    print(f"- Cross-Validation Mean: {results['cv_scores'].mean():.4f}")
    
    # Demo dự đoán
    print("\n" + "="*60)
    print("DEMO DỰ ĐOÁN")
    print("="*60)
    
    # Lấy một mẫu từ test set
    import pandas as pd
    sample = pd.DataFrame([dt.X_test[0]])
    prediction, probability = dt.predict(sample)
    
    print(f"\nMẫu dự đoán:")
    print(f"Kết quả dự đoán: {'Có bệnh tim' if prediction[0] == 1 else 'Không có bệnh tim'}")
    print(f"Xác suất không bệnh: {probability[0][0]:.4f}")
    print(f"Xác suất có bệnh: {probability[0][1]:.4f}")
    
    print(f"\n✓ Tất cả kết quả đã được lưu trong thư mục: {output_dir}")


if __name__ == "__main__":
    main()
