from collector import ProteinDataCollector
from processor import ProteinDataProcessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

if __name__ == "__main__":
    # Step 1: Collect data
    collector = ProteinDataCollector()
    df = collector.collect_ptm_data(ptm_type='CARBOHYD')
    print(f"Collected {len(df)} proteins. PTM positives: {df['ptm_label'].sum()}, negatives: {(df['ptm_label']==0).sum()}")

    # Step 2: Process data
    processor = ProteinDataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_data(df)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Step 3: Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Step 4: Evaluate
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred)) 