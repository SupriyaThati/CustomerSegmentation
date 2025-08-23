import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import matplotlib
matplotlib.use('Agg') 
from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

app = Flask(__name__)

try:
    df = pd.read_csv('Mall_Customers.csv')
    df = df.drop(columns=['CustomerID', 'Gender', 'Recency (days)', 'Frequency (visits)', 'Monetary ($)'], errors='ignore')
except FileNotFoundError:
    print("Error: Mall_Customers.csv not found in the project directory.")
   
    exit(1)

try:
    with open('kmeans_model.pkl', 'rb') as f:
        model_kmeans = pickle.load(f)
    with open('kmeans_scaler.pkl', 'rb') as f:
        scaler_kmeans = pickle.load(f)
    with open('meanshift_model.pkl', 'rb') as f:
        model_meanshift = pickle.load(f)
    with open('meanshift_scaler.pkl', 'rb') as f:
        scaler_meanshift = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: Missing model file - {e}")
 
    exit(1)
cluster_labels_kmeans = {
    0: {
        "name": "Balanced Spenders",
        "description": "Moderate-income customers who shop bi-weekly, spending sensibly with a focus on value-driven purchases. They balance quality and cost, avoiding impulsive buys.",
        "customer_type": "Middle-income families and professionals prioritizing value.",
        "customer_count": 83,
        "offers": "Offer 15% off regular purchases to encourage frequent visits, introduce a loyalty program with redeemable points, and provide bundle deals to enhance value."
    },
    1: {
        "name": "Affluent Spenders",
        "description": "High-income customers who shop weekly, spending generously on high-value experiences. They seek exclusivity and are willing to pay a premium.",
        "customer_type": "Wealthy individuals chasing premium purchases.",
        "customer_count": 39,
        "offers": "Provide 10% off high-value transactions, offer VIP membership perks, and host exclusive events to cater to their desire for prestige."
    },
    2: {
        "name": "Trendy Spenders",
        "description": "Low-income customers who shop weekly, spending heavily on trends driven by social media. They prioritize style despite budget constraints.",
        "customer_type": "Young adults and teens following trends.",
        "customer_count": 22,
        "offers": "Launch 20% off flash sales on new trends, offer limited-time bundles to create urgency, and engage through social media campaigns."
    },
    3: {
        "name": "Cautious Affluents",
        "description": "High-income customers who shop monthly, spending cautiously on durable, high-value items. They focus on long-term value over frequent purchases.",
        "customer_type": "Professionals and retirees seeking lasting value.",
        "customer_count": 35,
        "offers": "Offer 15% off high-quality purchases, provide extended payment plans, and promote exclusive deals to build trust."
    },
    4: {
        "name": "Frugal Spenders",
        "description": "Low-income customers who shop monthly, spending minimally on essentials. They prioritize savings and seek the best deals.",
        "customer_type": "Budget-conscious individuals and students.",
        "customer_count": 21,
        "offers": "Provide 20% off clearance deals, offer buy-one-get-one-free promotions on essentials, and distribute discount vouchers."
    }
}

cluster_labels_meanshift = {
    0: {
        "name": "Moderate Spenders",
        "description": "Balanced-income customers who shop bi-weekly, maintaining consistent spending on everyday needs. They value convenience and steady habits.",
        "customer_type": "Average families and workers with regular spending.",
        "customer_count": 78,
        "offers": "Offer 10% off regular purchases, provide loyalty points for frequent shopping, and run seasonal promotions."
    },
    1: {
        "name": "Conservative Elites",
        "description": "High-income customers who shop monthly, spending cautiously with a focus on value. They prioritize savings over frequent purchases.",
        "customer_type": "Affluent professionals seeking high-value buys.",
        "customer_count": 37,
        "offers": "Offer 15 off for high-value transactions, provide exclusive membership benefits, and promote extended warranties."
    },
    2: {
        "name": "Luxury Spenders",
        "description": "High-income customers who shop weekly, spending generously on premium purchases. They seek exclusivity and high-end experiences.",
        "customer_type": "Wealthy shoppers prioritizing status.",
        "customer_count": 39,
        "offers": "Provide 10% off premium purchases, offer VIP services like priority checkout, and host exclusive events."
    },
    3: {
        "name": "Thrifty Shoppers",
        "description": "Low-income customers who shop monthly, spending minimally on budget options. They focus on maximizing savings.",
        "customer_type": "Frugal individuals seeking affordability.",
        "customer_count": 23,
        "offers": "Offer 25% off clearance deals, provide bulk purchase discounts, and distribute weekly coupons."
    },
    4: {
        "name": "Trend-Driven Spenders",
        "description": "Low-income customers who shop weekly, spending heavily on trends influenced by social media. They prioritize style over savings.",
        "customer_type": "Young trend-followers with limited budgets.",
        "customer_count": 23,
        "offers": "Run 20% off flash sales on trendy items, offer limited-time deals, and engage through social media."
    }
}

def generate_cluster_plot(algo, cluster_id, income, score, labels, data, centers=None):
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    mask = labels == cluster_id
    cluster_name = (cluster_labels_kmeans.get(cluster_id, {"name": "Outliers"})["name"] if algo == 'kmeans' else
                    cluster_labels_meanshift.get(cluster_id, {"name": "Outliers"})["name"])
    plt.scatter(data[mask, 0], data[mask, 1], c=colors[cluster_id % len(colors)],
                label=cluster_name, alpha=0.6)

    if centers is not None:
        center = centers[cluster_id] if cluster_id < len(centers) else None
        if center is not None:
            plt.scatter([center[0]], [center[1]], c='black', marker='x', s=200, label='Center')
    plt.scatter([income], [score], c='red', marker='*', s=300, label='Your Customer')
    

    box_text = f"Group: {cluster_name}\nID: {cluster_id}"
    plt.text(0.95, 0.95, box_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=0.9),
             fontfamily='Arial')
    
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title(f'{algo_name_map[algo]} - {cluster_name}')
    plt.legend()
    
    plot_path = 'static/temp_cluster_plot.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    return plot_path

algo_name_map = {
    'kmeans': 'Standard Grouping',
    'meanshift': 'Flexible Grouping'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        income = float(request.form['income'])
        score = float(request.form['score'])

        age_str = request.form.get('age')
        
        if score < 1 or score > 100:
            return render_template('index.html', error="Spending Score must be between 1 and 100.")
        if income < 0:
            return render_template('index.html', error="Annual Income must be non-negative.")

        if age_str and age_str.strip(): 
            try:
                age = float(age_str)
                if age < 18 or age > 110:
                    return render_template('index.html', error="Age must be between 18 and 110.")
            except ValueError:
                return render_template('index.html', error="Please enter a valid numerical value for age.")

        algo = request.form['algorithm']

        plot_data = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
        input_df = pd.DataFrame([[income, score]], columns=['Annual Income (k$)', 'Spending Score (1-100)'])

        if algo == 'kmeans':
            data = scaler_kmeans.transform(input_df)
            cluster = model_kmeans.predict(data)[0]
            cluster_info = cluster_labels_kmeans.get(cluster, {
                "name": "Unknown",
                "description": "No category found.",
                "customer_type": "Unknown customer type.",
                "customer_count": 0,
                "offers": "Try different inputs."
            })
            algo_name = 'Standard Grouping'
            plot_labels = model_kmeans.labels_
            centers = scaler_kmeans.inverse_transform(model_kmeans.cluster_centers_)
            plot_path = generate_cluster_plot('kmeans', cluster, income, score, plot_labels, plot_data, centers)
        elif algo == 'meanshift':
            data = scaler_meanshift.transform(input_df)
            cluster = model_meanshift.predict(data)[0]
            cluster_info = cluster_labels_meanshift.get(cluster, {
                "name": "Unknown",
                "description": "No category found.",
                "customer_type": "Unknown customer type.",
                "customer_count": 0,
                "offers": "Try different inputs."
            })
            algo_name = 'Flexible Grouping'
            plot_labels = model_meanshift.labels_
            centers = scaler_meanshift.inverse_transform(model_meanshift.cluster_centers_)
            plot_path = generate_cluster_plot('meanshift', cluster, income, score, plot_labels, plot_data, centers)
        else:
            return render_template('index.html', error="Invalid clustering algorithm selected.")

        return render_template('result.html',
                               cluster=cluster,
                               name=cluster_info['name'],
                               description=cluster_info['description'],
                               customer_type=cluster_info['customer_type'],
                               customer_count=cluster_info['customer_count'],
                               offers=cluster_info['offers'],
                               algo_name=algo_name,
                               plot_file='temp_cluster_plot.png')
    except ValueError:
        return render_template('index.html', error="Please enter valid numerical values for annual income and spending score. Age must be a valid number if entered.")

@app.route('/download_plot')
def download_plot():
    plot_path = 'static/temp_cluster_plot.png'
    if os.path.exists(plot_path):
        return send_file(plot_path, as_attachment=True, download_name='cluster_plot.png')
    return "Plot not found", 404

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)  
    app.run(debug=True, port=5000)
