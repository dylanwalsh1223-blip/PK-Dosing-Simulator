import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import plotly.graph_objects as go

def oral_model(t, y, ka, k):
    A, C = y
    dA_dt = -ka * A
    dC_dt = ka * A - k * C
    return [dA_dt, dC_dt]

def iv_model(t, C, k):
    dC_dt = -k * C[0]
    return [dC_dt]

def compute_pk_curve_oral(ka, k, dose_amount, dose_interval, num_doses):
    times_list = []
    blood_list = []
    peak_per_dose = []
    current_state = [0, 0]

    for dose_number in range(num_doses):
        start_time = dose_number * dose_interval
        end_time = start_time + dose_interval
        t_segment = np.linspace(start_time, end_time, 50)

        current_state[0] = current_state[0] + dose_amount

        segment_solution = solve_ivp(oral_model, [start_time, end_time], current_state, args=(ka, k), t_eval=t_segment)

        times_list.extend(t_segment)
        blood_list.extend(segment_solution.y[1])
        peak_per_dose.append(max(segment_solution.y[1]))

        current_state = [segment_solution.y[0][-1], segment_solution.y[1][-1]]

    return times_list, blood_list, peak_per_dose

def compute_pk_curve_iv(k, dose_amount, dose_interval, num_doses):
    times_list = []
    blood_list = []
    peak_per_dose = []
    current_state = [0]

    for dose_number in range(num_doses):
        start_time = dose_number * dose_interval
        end_time = start_time + dose_interval
        t_segment = np.linspace(start_time, end_time, 50)

        current_state[0] = current_state[0] + dose_amount

        segment_solution = solve_ivp(iv_model, [start_time, end_time], current_state, args=(k,), t_eval=t_segment)

        times_list.extend(t_segment)
        blood_list.extend(segment_solution.y[0])
        peak_per_dose.append(max(segment_solution.y[0]))

        current_state = [segment_solution.y[0][-1]]

    return times_list, blood_list, peak_per_dose

def compute_pk_metrics(times_list, blood_list, k):
    times_array = np.array(times_list)
    blood_array = np.array(blood_list)

    max_concentration = np.max(blood_array)
    index_of_max = np.argmax(blood_array)
    time_of_max = times_array[index_of_max]

    half_life = np.log(2) / k

    area_under_curve = np.trapezoid(blood_array, times_array)

    return max_concentration, time_of_max, half_life, area_under_curve

def compute_loading_dose(target_concentration, volume_of_distribution):
    loading_dose = target_concentration * volume_of_distribution
    return loading_dose

def find_steady_state_dose(peak_per_dose, dose_interval):
    steady_state_dose_number = None

    for i in range(1, len(peak_per_dose)):
        previous_peak = peak_per_dose[i - 1]
        current_peak = peak_per_dose[i]
        percent_change = abs(current_peak - previous_peak) / current_peak

        if percent_change < 0.05:
            steady_state_dose_number = i + 1
            break

    if steady_state_dose_number is None:
        steady_state_time = None
    else:
        steady_state_time = (steady_state_dose_number - 1) * dose_interval

    return steady_state_dose_number, steady_state_time

def explain_single_drug(drug_name, route, max_concentration, therapeutic_min, therapeutic_max, kidney_function, dose_amount, dose_interval, num_doses, steady_state_dose_number, steady_state_time):
    lines = []

    lines.append("With " + str(num_doses) + " " + route + " doses of " + str(round(dose_amount, 1)) + " mg given every " + str(dose_interval) + " hours, " + drug_name + " reaches a peak blood concentration of " + str(round(max_concentration, 2)) + " mg/L.")

    if route == "IV":
        lines.append("Since this is given intravenously, the drug enters the bloodstream immediately with no absorption delay — the peak concentration happens right at the moment of each dose, unlike an oral dose which takes time to be absorbed first.")

    if max_concentration > therapeutic_max:
        lines.append("This peak is above the therapeutic window (max " + str(therapeutic_max) + " mg/L), meaning the drug is building up faster than the body can clear it — a sign of possible toxicity. This could be lowered by increasing the time between doses, reducing the dose amount, or both.")
    elif max_concentration < therapeutic_min:
        lines.append("This peak never reaches the therapeutic window (min " + str(therapeutic_min) + " mg/L), meaning the dose may be too small, or given too infrequently, to have a therapeutic effect. Increasing the dose or shortening the interval between doses could help.")
    else:
        lines.append("This peak stays within the therapeutic window (" + str(therapeutic_min) + "–" + str(therapeutic_max) + " mg/L), meaning this dosing schedule appears both safe and effective for this simplified model.")

    if kidney_function != "Normal":
        lines.append("Because kidney/liver function is set to '" + kidney_function + "', the drug is being eliminated more slowly than normal. This raises the risk of accumulation with repeated doses, which is why real-world dosing is often reduced for patients with impaired kidney or liver function.")

    if steady_state_dose_number is not None:
        lines.append("Steady state is reached around dose " + str(steady_state_dose_number) + " (about " + str(round(steady_state_time, 1)) + " hours in), meaning each new dose's peak stops climbing higher — the amount going in with each dose now roughly equals the amount being eliminated between doses.")
    else:
        lines.append("Steady state has not yet been reached within " + str(num_doses) + " doses — each dose is still building to a higher peak than the last. Try increasing the number of doses to see it level off.")

    return " ".join(lines)

def explain_comparison(drug_1_name, k_1, drug_2_name, k_2, max_1, max_2):
    if k_1 > k_2:
        faster_drug = drug_1_name
        slower_drug = drug_2_name
    else:
        faster_drug = drug_2_name
        slower_drug = drug_1_name

    explanation = faster_drug + " is eliminated from the body faster than " + slower_drug + " under these settings, so " + slower_drug + " stays in the bloodstream longer and is more likely to build up if doses are given close together. " + drug_1_name + " reached a peak of " + str(round(max_1, 2)) + " mg/L, while " + drug_2_name + " reached " + str(round(max_2, 2)) + " mg/L under the same dosing schedule."

    return explanation

st.set_page_config(page_title="PK Dosing Simulator", layout="wide")

drug_presets = {
    "Caffeine": {"ka": 1.0, "k": 0.14, "therapeutic_min": 2, "therapeutic_max": 10},
    "Ibuprofen": {"ka": 1.5, "k": 0.35, "therapeutic_min": 10, "therapeutic_max": 50},
    "Acetaminophen (Tylenol)": {"ka": 1.4, "k": 0.35, "therapeutic_min": 10, "therapeutic_max": 20},
    "Amoxicillin": {"ka": 1.1, "k": 0.46, "therapeutic_min": 4, "therapeutic_max": 30},
    "Metformin": {"ka": 0.9, "k": 0.12, "therapeutic_min": 1, "therapeutic_max": 4},
    "Atorvastatin": {"ka": 1.0, "k": 0.05, "therapeutic_min": 5, "therapeutic_max": 30},
    "Lisinopril": {"ka": 0.3, "k": 0.06, "therapeutic_min": 5, "therapeutic_max": 30},
    "Metoprolol": {"ka": 1.0, "k": 0.17, "therapeutic_min": 20, "therapeutic_max": 100},
    "Amlodipine": {"ka": 0.3, "k": 0.02, "therapeutic_min": 3, "therapeutic_max": 15},
    "Omeprazole": {"ka": 2.0, "k": 0.69, "therapeutic_min": 0.1, "therapeutic_max": 1},
    "Sertraline": {"ka": 0.4, "k": 0.03, "therapeutic_min": 10, "therapeutic_max": 50},
    "Warfarin": {"ka": 0.6, "k": 0.02, "therapeutic_min": 1, "therapeutic_max": 3},
    "Gabapentin": {"ka": 1.0, "k": 0.12, "therapeutic_min": 2, "therapeutic_max": 20},
    "Levothyroxine": {"ka": 0.15, "k": 0.004, "therapeutic_min": 0.005, "therapeutic_max": 0.012},
    "Prednisone": {"ka": 1.2, "k": 0.23, "therapeutic_min": 10, "therapeutic_max": 100},
    "Losartan": {"ka": 1.0, "k": 0.35, "therapeutic_min": 5, "therapeutic_max": 40},
    "Hydrochlorothiazide": {"ka": 0.5, "k": 0.07, "therapeutic_min": 5, "therapeutic_max": 40},
    "Simvastatin": {"ka": 1.2, "k": 0.28, "therapeutic_min": 5, "therapeutic_max": 30},
    "Azithromycin": {"ka": 0.4, "k": 0.01, "therapeutic_min": 0.1, "therapeutic_max": 1},
    "Ciprofloxacin": {"ka": 1.3, "k": 0.17, "therapeutic_min": 1, "therapeutic_max": 5},
    "Diazepam": {"ka": 0.5, "k": 0.016, "therapeutic_min": 0.1, "therapeutic_max": 1.5},
    "Alprazolam": {"ka": 1.0, "k": 0.063, "therapeutic_min": 0.01, "therapeutic_max": 0.1},
    "Fluoxetine": {"ka": 0.3, "k": 0.014, "therapeutic_min": 0.1, "therapeutic_max": 0.5},
    "Tramadol": {"ka": 1.0, "k": 0.12, "therapeutic_min": 0.1, "therapeutic_max": 0.5},
    "Doxycycline": {"ka": 0.9, "k": 0.043, "therapeutic_min": 1, "therapeutic_max": 5},
    "Custom (set your own values)": {"ka": 0.5, "k": 0.1, "therapeutic_min": 10, "therapeutic_max": 50}
}

kidney_function_multipliers = {
    "Normal": 1.0,
    "Mildly reduced": 0.75,
    "Moderately reduced": 0.5,
    "Severely reduced": 0.25
}

st.title("Pharmacokinetic Dosing Simulator")
st.write("Choose a drug preset or set your own absorption and elimination values, then adjust the dosing schedule to see how blood concentration changes over time.")
st.caption("Educational approximations for demonstration purposes only — not clinical dosing guidance.")

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Route of administration")
    route = st.radio("Route", ["Oral", "IV"], horizontal=True)

    st.subheader("Drug and dosing")

    selected_drug = st.selectbox("Drug", list(drug_presets.keys()))
    preset = drug_presets[selected_drug]

    if selected_drug == "Custom (set your own values)":
        ka = st.slider("Absorption rate (ka)", min_value=0.1, max_value=2.0, value=preset["ka"])
        k = st.slider("Elimination rate (k)", min_value=0.01, max_value=0.5, value=preset["k"])
    else:
        ka = preset["ka"]
        k = preset["k"]
        st.write("Absorption rate (ka):", ka)
        st.write("Elimination rate (k):", k)

    if route == "IV":
        st.caption("IV bypasses absorption — the drug enters the bloodstream immediately, so ka is not used.")

    st.subheader("Patient factors")

    weight_kg = st.slider("Patient weight (kg)", min_value=10, max_value=150, value=70)

    use_weight_dosing = st.checkbox("Scale dose by weight (mg/kg)")

    if use_weight_dosing:
        dose_per_kg = st.slider("Dose per kg (mg/kg)", min_value=0.5, max_value=10.0, value=1.5)
        dose_amount = dose_per_kg * weight_kg
        st.write("Calculated dose:", round(dose_amount, 1), "mg")
    else:
        dose_amount = st.slider("Dose amount (mg)", min_value=50, max_value=500, value=100)

    kidney_function = st.selectbox("Kidney/liver function", list(kidney_function_multipliers.keys()))
    kidney_multiplier = kidney_function_multipliers[kidney_function]
    k_adjusted = k * kidney_multiplier

    if kidney_function != "Normal":
        st.write("Adjusted elimination rate (k):", round(k_adjusted, 4))

    dose_interval = st.slider("Hours between doses", min_value=2, max_value=24, value=8)
    num_doses = st.slider("Number of doses", min_value=1, max_value=20, value=6)

    st.subheader("Loading dose calculator")
    st.caption("A loading dose is a larger first dose used to reach a target concentration immediately, instead of waiting several half-lives to build up.")

    target_concentration = st.slider("Target concentration (mg/L)", min_value=0.1, max_value=50.0, value=float(preset["therapeutic_min"]))
    volume_of_distribution = st.slider("Volume of distribution, Vd (L)", min_value=5.0, max_value=100.0, value=40.0)

    loading_dose = compute_loading_dose(target_concentration, volume_of_distribution)

    st.metric("Calculated loading dose", str(round(loading_dose, 1)) + " mg")
    st.caption("Loading dose = target concentration × volume of distribution")

    st.subheader("Compare drugs")
    compare_mode = st.checkbox("Compare against a second drug")

    if compare_mode:
        selected_drug_2 = st.selectbox("Drug 2", list(drug_presets.keys()), index=1)
        preset_2 = drug_presets[selected_drug_2]
        ka_2 = preset_2["ka"]
        k_2 = preset_2["k"]
        k_2_adjusted = k_2 * kidney_multiplier

    st.subheader("Therapeutic window (Drug 1)")
    therapeutic_min = st.slider("Minimum (mg/L)", min_value=0.0, max_value=50.0, value=float(preset["therapeutic_min"]))
    therapeutic_max = st.slider("Maximum (mg/L)", min_value=0.0, max_value=100.0, value=float(preset["therapeutic_max"]))

if route == "Oral":
    times_list, blood_list, peak_per_dose = compute_pk_curve_oral(ka, k_adjusted, dose_amount, dose_interval, num_doses)
else:
    times_list, blood_list, peak_per_dose = compute_pk_curve_iv(k_adjusted, dose_amount, dose_interval, num_doses)

if compare_mode:
    if route == "Oral":
        times_list_2, blood_list_2, peak_per_dose_2 = compute_pk_curve_oral(ka_2, k_2_adjusted, dose_amount, dose_interval, num_doses)
    else:
        times_list_2, blood_list_2, peak_per_dose_2 = compute_pk_curve_iv(k_2_adjusted, dose_amount, dose_interval, num_doses)

max_concentration, time_of_max, half_life, area_under_curve = compute_pk_metrics(times_list, blood_list, k_adjusted)
steady_state_dose_number, steady_state_time = find_steady_state_dose(peak_per_dose, dose_interval)

with col2:
    st.subheader(selected_drug + " (" + route + ") — blood concentration over time")

    figure = go.Figure()

    figure.add_shape(
        type="rect",
        x0=min(times_list), x1=max(times_list),
        y0=therapeutic_min, y1=therapeutic_max,
        fillcolor="lightgreen", opacity=0.3, line_width=0
    )

    figure.add_trace(go.Scatter(x=times_list, y=blood_list, mode="lines", name=selected_drug, line=dict(color="royalblue", width=3)))

    if compare_mode:
        figure.add_trace(go.Scatter(x=times_list_2, y=blood_list_2, mode="lines", name=selected_drug_2, line=dict(color="darkorange", width=3)))

    if steady_state_time is not None:
        figure.add_vline(x=steady_state_time, line_dash="dash", line_color="gray")
        figure.add_annotation(x=steady_state_time, y=max(blood_list), text="Steady state reached", showarrow=True, arrowhead=1)

    figure.update_layout(
        xaxis_title="Time (hours)",
        yaxis_title="Concentration (mg/L)",
        showlegend=compare_mode,
        height=450
    )

    st.plotly_chart(figure, use_container_width=True)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Cmax (peak)", str(round(max_concentration, 2)) + " mg/L")
    metric_col2.metric("Tmax (time of peak)", str(round(time_of_max, 1)) + " hr")
    metric_col3.metric("Half-life", str(round(half_life, 1)) + " hr")
    metric_col4.metric("AUC", str(round(area_under_curve, 1)) + " mg·hr/L")

    if max_concentration > therapeutic_max:
        st.error("Exceeds therapeutic window — toxicity risk")
    elif max_concentration < therapeutic_min:
        st.error("Never reaches therapeutic window — may be ineffective")
    else:
        st.success("Stays within therapeutic window")

    st.subheader("What this means")

    if compare_mode:
        max_concentration_2 = max(blood_list_2)
        st.info(explain_comparison(selected_drug, k_adjusted, selected_drug_2, k_2_adjusted, max_concentration, max_concentration_2))
    else:
        st.info(explain_single_drug(selected_drug, route, max_concentration, therapeutic_min, therapeutic_max, kidney_function, dose_amount, dose_interval, num_doses, steady_state_dose_number, steady_state_time))