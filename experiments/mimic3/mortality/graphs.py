

import csv
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt 


explainer_to_code={
"deep_lift":0,
"dyna_mask":1,
"integrated_gradients":3,
"augmented_occlusion":4,
"occlusion":5,
"retain":6,
"lime":7,
"gradient_shap":8,
"extremal_mask":9
}

format_explainer={
"deep_lift":"DeepLift",
"dyna_mask":"DynaMask",
"integrated_gradients":"Integrated Gradients",
"augmented_occlusion":"Augmented Occlusion",
"occlusion":"Occlusion",
"retain":"Retain",
"lime":"Lime",
"gradient_shap":"GradientShap",
"extremal_mask":"Extremal Mask"
}

code_to_explainer={}
for k,v in explainer_to_code.items():
    code_to_explainer[v]=k

baseline_to_code={
"Zeros":0,
"Average":1,
}
code_to_baseline={
0:"Zeros",
1:"Average"
}
columns="Seed,Fold,Baseline,Topk,Explainer,Lambda_1,Lambda_2,Accuracy,Comprehensiveness,Cross Entropy,Log Odds,Sufficiency".split(",")
        





def get_mimic_data(filename):
    total_raw_values=None
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            raw_values = np.array([ row["Seed"], row["Fold"], baseline_to_code[row["Baseline"]], row["Topk"], explainer_to_code[row["Explainer"]], 
                row["Lambda_1"],row["Lambda_2"],row["Accuracy"],row["Comprehensiveness"],row["Cross Entropy"],row["Log Odds"],
                row["Sufficiency"]]).reshape(1,-1).astype(np.float32)
            total_raw_values = raw_values if total_raw_values is None else np.concatenate((total_raw_values, raw_values))
    return total_raw_values 
    

def print_mimic_table(total_raw_values, human_readable=True):
    values=pd.DataFrame(total_raw_values,columns=columns)  
    mask1 = values['Topk']==0.2
    mask2= values["Baseline"] == baseline_to_code["Average"]
    ave_02_vals = values[mask1 & mask2].sort_values("Baseline")

    mean_ave_02_vals=ave_02_vals.groupby("Explainer").mean()
    mean_ave_02_vals=mean_ave_02_vals.reset_index()

    std_ave_02_vals=ave_02_vals.groupby("Explainer").std()
    std_ave_02_vals=std_ave_02_vals.reset_index()

    plus_minus =  " +/- "if human_readable else " $\pm$ "
    row_connector =  " "if human_readable else " & "
    

    for i in range(mean_ave_02_vals.shape[0]):
        mean_row=mean_ave_02_vals.iloc[i]
        std_row=std_ave_02_vals.iloc[i]
        explainer_name = format_explainer[code_to_explainer[std_row["Explainer"]]]
        table_row_vals = ["      "+explainer_name]
        cols_to_include = ["Accuracy","Comprehensiveness","Cross Entropy","Sufficiency"]
        
        f_styles = ["{:.3f}","{:.2e}","{:.2f}","{:.2e}"]
        for i in range(len(cols_to_include)):
            col_name = cols_to_include[i]
            f_style = f_styles[i]
            mean=f_style.format( mean_row[col_name])
            std="{:.3f}".format( std_row[col_name])

            if(human_readable):
                table_row_vals.append(col_name+" "+mean+",")
            else:
                table_row_vals.append(   mean+plus_minus+std)
        table_row=row_connector.join(table_row_vals)+" \n"
        if(explainer_name=="Extremal Mask"):
            print("      \hline")
        print(table_row)

def create_mimic_graph(total_raw_values,baseline):
    values=pd.DataFrame(total_raw_values,columns=columns)  
    x=np.arange(0,0.61,0.1)
    values=values[   (values['Baseline']==baseline_to_code[baseline])]
    group_keys = ['Explainer', 'Topk']
    values = values.groupby(group_keys, as_index=False)
    mean_vals = values.mean()

    std_vals = values.std()
    std_vals = std_vals[group_keys+["Cross Entropy"]]
    std_vals=std_vals.rename(columns={"Cross Entropy": "Std Cross Entropy"})

    values=pd.merge(mean_vals, std_vals, on = group_keys)


    color_mapping = {
        "deep_lift":"b",
        "occlusion":"g",
        "extremal_mask":"r",
        "dyna_mask":'tab:orange'
    }

    for explainer, code in explainer_to_code.items():
        if(explainer in color_mapping.keys()):
            explainer_vals=values[values['Explainer']==code]
            x=explainer_vals['Topk'].to_numpy()
            y=explainer_vals['Cross Entropy'].to_numpy()
            error =explainer_vals['Std Cross Entropy'].to_numpy()    
            plt.plot(x, y, label = explainer, color=color_mapping[explainer]) 
            plt.fill_between(x, y-error, y+error, alpha=0.1, color=color_mapping[explainer])
    plt.legend() 
    plt.show()



if __name__ == "__main__":
    total_raw_values=get_mimic_data("compiled.csv")
    print_mimic_table(total_raw_values)
    create_mimic_graph(total_raw_values, "Average")
    create_mimic_graph(total_raw_values, "Zeros")
