#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: visualize
Author: 钱隆
Date: 2024-12-08
Email: neymarql0614@gmail.com

Modification History:
    - Date: 2024-12-12
      Author: 钱隆
"""
import json

from matplotlib import pyplot as plt


def visualize_results():
    import glob

    results = {}
    result_files = glob.glob("./results_ddp/*/results.json")
    for file in result_files:
        model_name = file.split('/')[2]
        with open(file, 'r') as f:
            data = json.load(f)
            results[model_name] = data['fine_tuned_test_acc']

    # 可视化测试准确率
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    accuracies = [results[model] for model in models]
    plt.bar(models, accuracies, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    plt.title('Model Comparison on Test Accuracy')
    plt.ylim(0, 1)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center')
    plt.show()

visualize_results()