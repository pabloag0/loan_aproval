import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sklearn.cluster import KMeans



def show_dataset_info(df, plot=False):
    ages = pd.cut(df['person_age'], bins=[0, 18, 30, 50, 70, np.inf], labels=['minor', 'junior', 'mid_age', 'senior', 'elderly']).value_counts() 
    paridad = df['person_gender'].value_counts()
    studies = df['person_education'].value_counts()
    income = pd.cut(df['person_income'], bins=[0, 20000, 50000, 100000, 200000, np.inf], labels=['low', 'lower_middle', 'middle', 'upper_middle', 'high']).value_counts()
    experience = pd.cut(df['person_emp_exp'], bins=[0, 1, 3, 5, 10, np.inf], labels=['<1 year', '1-3 years', '3-5 years', '5-10 years', '>10 years']).value_counts()
    home = df['person_home_ownership'].value_counts()
    amount = pd.cut(df['loan_amnt'], bins=[0, 5000, 10000, 20000, 50000, np.inf], labels=['very_low', 'low', 'medium', 'high', 'very_high']).value_counts()
    intention = df['loan_intent'].value_counts()
    interest = pd.cut(df['loan_int_rate'], bins=[0, 5, 10, 15, 20, np.inf], labels=['very_low', 'low', 'medium', 'high', 'very_high']).value_counts()
    percent_income = pd.cut(df['loan_percent_income'], bins=[0, 10, 20, 30, 40, np.inf], labels=['very_low', 'low', 'medium', 'high', 'very_high']).value_counts()
    credit_history_lenght = pd.cut(df['cb_person_cred_hist_length'], bins=[0, 1, 3, 5, 10, np.inf], labels=['<1 year', '1-3 years', '3-5 years', '5-10 years', '>10 years']).value_counts()
    credit_score = pd.cut(df['credit_score'], bins=[0, 300, 600, 700, 800, np.inf], labels=['very_low', 'low', 'medium', 'high', 'very_high']).value_counts()
    previous_loan = df['previous_loan_defaults_on_file'].value_counts()
    status = df['loan_status'].value_counts()

    print('Loan Dataset Information:')
    print('-----------------------------')
    print(ages)
    print('-----------------------------')
    print(paridad)
    print('-----------------------------')
    print(studies)
    print('-----------------------------')
    print(income)
    print('-----------------------------') 
    print(experience)
    print('-----------------------------') 
    print(home)
    print('-----------------------------')
    print(amount)
    print('-----------------------------')
    print(intention)
    print('-----------------------------')
    print(interest)
    print('-----------------------------')
    print(percent_income)
    print('-----------------------------')
    print(credit_history_lenght)
    print('-----------------------------')
    print(credit_score)
    print('-----------------------------')
    print(previous_loan)
    print('-----------------------------')
    print(status)
    print('-----------------------------')

    if plot:
        data_to_plot = {
            'Age': ages,
            'Gender': paridad,
            'Education': studies,
            'Income': income,
            'Experience': experience,
            'Home Ownership': home,
            'Loan Amount': amount,
            'Loan Intent': intention,
            'Interest Rate': interest,
            'Percent Income': percent_income,
            'Credit History': credit_history_lenght,
            'Credit Score': credit_score,
            'Previous Default': previous_loan,
            'Loan Status': status,
        }

        save_dir = 'C:/Users/pablo/OneDrive - Universidad Complutense de Madrid (UCM)/Uni/3º/2º/AA/workspace/Proyecto/results/figures/eda/'
        os.makedirs(save_dir, exist_ok=True)

        # --- Figura 1: todas las distribuciones individuales ---
        n = len(data_to_plot)
        ncols = 4
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axes = axes.flatten()

        for i, (title, series) in enumerate(data_to_plot.items()):
            series.plot(kind='bar', ax=axes[i], title=title)
            axes[i].tick_params(axis='x', rotation=45)

        # Ocultar ejes sobrantes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'all_distributions.png'), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)

        # --- Figura 2: Gender vs Loan Status ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Gender vs Loan Status', fontsize=14)

        gender_loan = df.groupby(['person_gender', 'loan_status']).size().unstack(fill_value=0)
        gender_loan.plot(kind='bar', ax=axes[0], title='Absolute Count')
        axes[0].tick_params(axis='x', rotation=0)
        axes[0].set_xlabel('Gender')
        axes[0].set_ylabel('Count')

        gender_loan_pct = gender_loan.div(gender_loan.sum(axis=1), axis=0) * 100
        gender_loan_pct.plot(kind='bar', ax=axes[1], title='Approval Rate (%)')
        axes[1].tick_params(axis='x', rotation=0)
        axes[1].set_xlabel('Gender')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gender_vs_loan_status.png'), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)

# CLUSTERING
def kmeans(X):
    # No se puede aplazar más, toca clustering
    model = KMeans(n_clusters=10, random_state=42, n_init=10)
    clusters = model.fit_predict(X)
    return model, clusters

if __name__ == "__main__":
    pass
