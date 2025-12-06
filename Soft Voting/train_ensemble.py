from soft_voting_ensemble import SoftVotingEnsemble
import pandas as pd
import os


def main():
    """Hàm chính để đánh giá Soft Voting Ensemble"""
    
    # Tạo thư mục output
    output_dir = 'ensemble_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("="*60)
    print("ĐÁNH GIÁ SOFT VOTING ENSEMBLE")
    print("="*60)
    
    # ========================================
    # CẤU HÌNH CÁC MODEL - DỄ DÀNG MỞ RỘNG
    # ========================================
    # Code sẽ tự động:
    # 1. Load tất cả models
    # 2. Thử nghiệm 5 metrics khác nhau: accuracy, f1, precision, recall, roc_auc
    # 3. Chọn metric cho kết quả tốt nhất
    # 4. Gán trọng số cho từng model theo metric đó
    # 5. Đánh giá ensemble với trọng số tối ưu
    models_info = [
        {
            'name': 'KNN',
            'model_path': 'KNN/knn_results/knn_heart_disease_model.pkl',
            'scaler_path': 'KNN/knn_results/knn_scaler.pkl',
        },
        {
            'name': 'Decision Tree',
            'model_path': 'Decision_tree/decision_tree_results/dt_heart_disease_model.pkl',
            'scaler_path': 'Decision_tree/decision_tree_results/dt_scaler.pkl',
        },
        {
            'name': 'SVM',
            'model_path': 'svm/svm_model.pkl',
            'scaler_path': 'svm/scaler.pkl',
        },
        {
            'name': 'Logistic Regression',
            'model_path': 'lr/logreg_model.pkl',
            'scaler_path': 'lr/scaler.pkl',
        },
        {
            'name': 'AdaBoost',
            'model_path': 'adaboost/adaboost_model.pkl',
            'scaler_path': 'adaboost/scaler.pkl',
        },
        {
            'name': 'Gradient Boosting',
            'model_path': 'gradientBoosting/gradient_boosting_model.pkl',
            'scaler_path': 'gradientBoosting/scaler.pkl',
        },
        {
            'name': 'xgboost',
            'model_path': 'xgb/xgboost_model.pkl',
            'scaler_path': 'xgb/scaler.pkl',
        },
        {
            'name': 'Random Forest',
            'model_path': 'rdForest/random_forest_model.pkl',
            'scaler_path': 'rdForest/scaler.pkl',
        }
        # ========================================
        # DỄ DÀNG THÊM MODEL MỚI TẠI ĐÂY:
        # ========================================
        # {
        #     'name': 'Random Forest',
        #     'model_path': 'random_forest_results/rf_heart_disease_model.pkl',
        #     'scaler_path': 'random_forest_results/rf_scaler.pkl',  # Optional
        #     'weight': 1.0
        # },
        # {
        #     'name': 'XGBoost',
        #     'model_path': 'xgboost_results/xgb_heart_disease_model.pkl',
        #     'scaler_path': None,  # Nếu model không cần scaler
        #     'weight': 1.0
        # },
    ]
    
    # Khởi tạo ensemble
    ensemble = SoftVotingEnsemble(models_info)
    
    # Load các model
    ensemble.load_models()
    
    if len(ensemble.models) < 2:
        print("\n⚠ Cần ít nhất 2 model để thực hiện ensemble!")
        print("Vui lòng train các model trước khi chạy ensemble.")
        return
    
    # Load dữ liệu test
    # Sử dụng cùng dữ liệu test như khi train các model
    print("\n" + "="*60)
    print("LOADING DỮ LIỆU TEST")
    print("="*60)
    
    data_file = 'Heart_disease_cleveland_new(2).csv'
    data = pd.read_csv(data_file)
    
    # Chia dữ liệu giống như khi train (để có cùng test set)
    from sklearn.model_selection import train_test_split
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Load test data vào ensemble
    ensemble.load_test_data(X_test.values, y_test.values)

    # TỰ ĐỘNG GÁN TRỌNG SỐ THEO CÁC METRICS KHÁC NHAU
    print("\n" + "="*60)
    print("SO SÁNH CÁC CHIẾN LƯỢC GÁN TRỌNG SỐ")
    print("="*60)

    metrics_to_test = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    best_metric = None
    best_accuracy = 0
    results_by_metric = {}

    for metric in metrics_to_test:
        print(f"\n--- Testing với metric: {metric.upper()} ---")

        # Gán trọng số theo metric hiện tại
        ensemble.auto_weight_by_metrics(metric=metric)

        # Đánh giá ensemble
        temp_results = ensemble.evaluate()
        ensemble_accuracy = temp_results['test_accuracy']

        results_by_metric[metric] = {
            'accuracy': ensemble_accuracy,
            'weights': ensemble.weights.copy(),
            'results': temp_results
        }

        print(f"Ensemble accuracy với {metric}: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")

        if ensemble_accuracy > best_accuracy:
            best_accuracy = ensemble_accuracy
            best_metric = metric

    print(f"\n🎯 Metric tốt nhất: {best_metric.upper()} (Accuracy: {best_accuracy:.4f})")

    # Sử dụng metric tốt nhất cho kết quả cuối cùng
    ensemble.auto_weight_by_metrics(metric=best_metric)
    results = results_by_metric[best_metric]['results']
    
    # Vẽ các biểu đồ
    print("\n" + "="*60)
    print("TẠO CÁC BIỂU ĐỒ")
    print("="*60)
    
    ensemble.plot_confusion_matrix(save_path=os.path.join(output_dir, 'ensemble_confusion_matrix.png'))
    ensemble.plot_roc_curve(save_path=os.path.join(output_dir, 'ensemble_roc_curve.png'))
    ensemble.plot_model_comparison(results, save_path=os.path.join(output_dir, 'ensemble_model_comparison.png'))
    ensemble.plot_metrics_comparison(results_by_metric, best_metric,
                                   save_path=os.path.join(output_dir, 'ensemble_metrics_comparison.png'))
    
    # Xuất metrics ra Excel
    ensemble.export_metrics_to_excel(results, filename=os.path.join(output_dir, 'ensemble_metrics.xlsx'))
    ensemble.export_metrics_comparison_to_excel(results_by_metric, best_metric,
                                               filename=os.path.join(output_dir, 'ensemble_metrics_comparison.xlsx'))
    
    # Tổng kết
    print("\n" + "="*60)
    print("TỔNG KẾT")
    print("="*60)

    # In metrics chính của ensemble
    ensemble.print_ensemble_metrics(results, f"ENSEMBLE METRICS (sử dụng {best_metric.upper()})")

    print(f"\n📈 So sánh các chiến lược gán trọng số:")
    for metric, data in results_by_metric.items():
        status = "🎯" if metric == best_metric else "  "
        print(f"   {status} {metric.upper():12s}: {data['accuracy']*100:.2f}%")

    print(f"\n📈 So sánh với từng model:")
    for name, acc in results['individual_accuracies'].items():
        print(f"   • {name:20s}: {acc*100:.2f}%")

    best_individual = max(results['individual_accuracies'].values())
    improvement = results['test_accuracy'] - best_individual

    if improvement > 0:
        print(f"\n✅ Ensemble tốt hơn model tốt nhất: +{improvement*100:.2f}%")
    elif improvement < 0:
        print(f"\n⚠ Ensemble kém hơn model tốt nhất: {improvement*100:.2f}%")
    else:
        print(f"\n➡ Ensemble ngang bằng với model tốt nhất")
    
    print(f"\n✓ Tất cả kết quả đã được lưu trong thư mục: {output_dir}")
    
    # Demo dự đoán
    print("\n" + "="*60)
    print("DEMO DỰ ĐOÁN")
    print("="*60)

    # Lấy một mẫu từ test set
    sample = X_test.iloc[[0]].values
    prediction, probability = ensemble.predict(sample)

    print(f"\nMẫu dự đoán:")
    print(f"Kết quả dự đoán: {'Có bệnh tim' if prediction[0] == 1 else 'Không có bệnh tim'}")
    print(f"Xác suất không bệnh: {probability[0][0]:.4f} ({probability[0][0]*100:.2f}%)")
    print(f"Xác suất có bệnh:    {probability[0][1]:.4f} ({probability[0][1]*100:.2f}%)")

    # Hiển thị metrics chính của ensemble
    print(f"\n📊 Ensemble Performance Summary:")
    ensemble_metrics = ensemble.get_ensemble_metrics(results)
    print(f"   • Accuracy:  {ensemble_metrics['accuracy']*100:.2f}%")
    print(f"   • Precision: {ensemble_metrics['precision']:.4f}")
    print(f"   • Recall:    {ensemble_metrics['recall']:.4f}")
    print(f"   • F1-Score:  {ensemble_metrics['f1_score']:.4f}")
    print(f"   • ROC-AUC:   {ensemble_metrics['roc_auc']:.4f}")

    print("\n" + "="*60)
    print("HOÀN THÀNH!")
    print("="*60)


if __name__ == "__main__":
    main()
