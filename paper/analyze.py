import pandas as pd
import numpy as np
import os

# File paths
margo1_csv = r"c:\Users\thisi\Desktop\ErfanMhp\publications\MARGO\result_final_v2\MARGO\v2v+energy\10\20251125_133001\MARGO1.csv"
margo2_csv = r"c:\Users\thisi\Desktop\ErfanMhp\publications\MARGO\result_final_v2\MARGO\v2v+energy\12\20251125_123045\MARGO2.csv"
mrlco1_csv = r"c:\Users\thisi\Desktop\ErfanMhp\publications\MARGO\result_final_v2\MRLCO\10\MRLCO1.csv"
mrlco2_csv = r"c:\Users\thisi\Desktop\ErfanMhp\publications\MARGO\result_final_v2\MRLCO\12\MRLCO2.csv"
margo1_xlsx = r"c:\Users\thisi\Desktop\ErfanMhp\publications\MARGO\result_final_v2\MARGO\v2v+energy\10\detailed_iterations\iteration_100_detailed_MARGO_1.xlsx"
margo2_xlsx = r"c:\Users\thisi\Desktop\ErfanMhp\publications\MARGO\result_final_v2\MARGO\v2v+energy\12\detailed_iterations\iteration_100_detailed_MARGO_2.xlsx"

out_dir = "analysis_tables"
os.makedirs(out_dir, exist_ok=True)

def analyze_csv(path, name):
    df = pd.read_csv(path)
    assert len(df) == 101, f"{name} does not have 101 rows"
    
    final_row = df.iloc[100]
    last20 = df.iloc[81:101]
    
    greedy_lat = last20['greedy_latencies'].mean()
    greedy_ene = last20['greedy_energy'].mean()
    
    last20_lat = last20['average_latency'].mean()
    last20_ene = last20['average_energy'].mean()
    
    lat_impr = (greedy_lat - last20_lat) / greedy_lat * 100
    ene_impr = (greedy_ene - last20_ene) / greedy_ene * 100
    comp_impr = 0.5 * lat_impr + 0.5 * ene_impr
    
    return {
        "Task": name.split()[0],
        "Method": name.split()[1],
        "Final Lat.": final_row['average_latency'],
        "Last-20 Lat.": last20_lat,
        "Lat. Impr.": f"{lat_impr:.2f}%",
        "Final Energy": final_row['average_energy'],
        "Last-20 Energy": last20_ene,
        "Energy Impr.": f"{ene_impr:.2f}%",
        "Lat. Std.": last20['average_latency'].std(),
        "Energy Std.": last20['average_energy'].std(),
        "Comp. Impr.": f"{comp_impr:.2f}%"
    }

res = []
res.append(analyze_csv(margo1_csv, "T1 MARGO"))
res.append(analyze_csv(mrlco1_csv, "T1 MRLCO-style"))
res.append(analyze_csv(margo2_csv, "T2 MARGO"))
res.append(analyze_csv(mrlco2_csv, "T2 MRLCO-style"))

df_perf = pd.DataFrame(res)
df_perf.to_csv(os.path.join(out_dir, "main_performance_stability.csv"), index=False)

def analyze_xlsx(path, task_name):
    df = pd.read_excel(path)
    assert len(df) == 2000, f"{task_name} does not have 2000 rows"
    
    # Action distribution
    actions = ['Local', 'MEC', 'V2V']
    action_stats = []
    for act in actions:
        sub = df[df['Action_Name'] == act]
        action_stats.append({
            "Task": task_name,
            "Action": act,
            "Decisions": len(sub),
            "Share": f"{len(sub)/2000*100:.2f}%",
            "Avg Latency": sub['Latency'].mean(),
            "Avg Energy": sub['Energy_Consumption'].mean(),
            "Avg Depth": sub['Task_Depth'].mean(),
            "Avg Pred.": sub['Num_Predecessors'].mean(),
            "Avg Succ.": sub['Num_Successors'].mean()
        })
        
    # Structure conditioned
    struct_stats = []
    
    def get_struct_row(name, sub_df):
        if len(sub_df) == 0:
            return {"Task": task_name, "Structural Group": name, "Nodes": 0, "Local": "0.00%", "MEC": "0.00%", "V2V": "0.00%"}
        return {
            "Task": task_name,
            "Structural Group": name,
            "Nodes": len(sub_df),
            "Local": f"{len(sub_df[sub_df['Action_Name']=='Local'])/len(sub_df)*100:.2f}%",
            "MEC": f"{len(sub_df[sub_df['Action_Name']=='MEC'])/len(sub_df)*100:.2f}%",
            "V2V": f"{len(sub_df[sub_df['Action_Name']=='V2V'])/len(sub_df)*100:.2f}%"
        }
        
    struct_stats.append(get_struct_row("Depth 0", df[df['Task_Depth'] == 0]))
    struct_stats.append(get_struct_row("Depth 1", df[df['Task_Depth'] == 1]))
    struct_stats.append(get_struct_row("Depth >=2", df[df['Task_Depth'] >= 2]))
    struct_stats.append(get_struct_row("Exit nodes", df[df['Num_Successors'] == 0]))
    struct_stats.append(get_struct_row("Successors 1-2", df[(df['Num_Successors'] >= 1) & (df['Num_Successors'] <= 2)]))
    struct_stats.append(get_struct_row("Successors >=3", df[df['Num_Successors'] >= 3]))
    
    # Graph level
    graph_stats = []
    makespans = df.groupby('Graph_ID')['Finish_Time'].max()
    energies = df.groupby('Graph_ID')['Energy_Consumption'].sum()
    act_counts = df.groupby('Graph_ID')['Action_Name'].value_counts().unstack(fill_value=0)
    for act in actions:
        if act not in act_counts.columns:
            act_counts[act] = 0
            
    v2v_min = act_counts['V2V'].min()
    v2v_max = act_counts['V2V'].max()
    
    graph_stats.append({
        "Task": task_name,
        "Avg Makespan": makespans.mean(),
        "Avg Energy": energies.mean(),
        "Avg Local Actions": act_counts['Local'].mean(),
        "Avg MEC Actions": act_counts['MEC'].mean(),
        "Avg V2V Actions": act_counts['V2V'].mean(),
        "V2V Range": f"{v2v_min}-{v2v_max}"
    })
    
    return action_stats, struct_stats, graph_stats

act1, str1, grp1 = analyze_xlsx(margo1_xlsx, "T1")
act2, str2, grp2 = analyze_xlsx(margo2_xlsx, "T2")

pd.DataFrame(act1 + act2).to_csv(os.path.join(out_dir, "node_level_policy_behavior.csv"), index=False)
pd.DataFrame(str1 + str2).to_csv(os.path.join(out_dir, "structure_conditioned_actions.csv"), index=False)
pd.DataFrame(grp1 + grp2).to_csv(os.path.join(out_dir, "graph_level_summary.csv"), index=False)

print("Analysis complete.")
