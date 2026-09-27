import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

def analytical_solution(t, C0, k):
    return C0 * np.exp(-k * t)

def ode_model(t, C, k):
    return -k * C

def oral_model(t, y, ka, k):
    A, C = y
    dA_dt = -ka * A
    dC_dt = ka * A - k * C
    return [dA_dt, dC_dt]

def saturable_model(t, C, Vmax, Km):
    dC_dt = -(Vmax * C[0]) / (Km + C[0])
    return [dC_dt]

t = np.linspace(0, 24, 100)
C = analytical_solution(t, 100, 0.1)
solution = solve_ivp(ode_model, [0, 24], [100], args=(0.1,), t_eval=t)
solution_oral = solve_ivp(oral_model, [0, 24], [100, 0], args=(0.5, 0.1), t_eval=t)

dose_interval = 8
num_doses = 3
dose_amount = 100

times_list = []
gut_list = []
blood_list = []

current_state = [0, 0]

for dose_number in range(num_doses):
    start_time = dose_number * dose_interval
    end_time = start_time + dose_interval
    t_segment = np.linspace(start_time, end_time, 50)

    current_state[0] = current_state[0] + dose_amount

    segment_solution = solve_ivp(oral_model, [start_time, end_time], current_state, args=(0.5, 0.1), t_eval=t_segment)

    times_list.extend(t_segment)
    gut_list.extend(segment_solution.y[0])
    blood_list.extend(segment_solution.y[1])

    current_state = [segment_solution.y[0][-1], segment_solution.y[1][-1]]

solution_saturable = solve_ivp(saturable_model, [0, 24], [100], args=(20, 30), t_eval=t)

real_time_data = np.array([0.5, 1, 2, 4, 6, 8, 10, 12])
real_concentration_data = np.array([8.5, 9.6, 8.2, 6.0, 4.3, 3.1, 2.2, 1.6])

def oral_fit_function(t_input, ka, k):
    result = solve_ivp(oral_model, [0, max(t_input)], [10, 0], args=(ka, k), t_eval=t_input)
    return result.y[1]

best_params, covariance = curve_fit(oral_fit_function, real_time_data, real_concentration_data, p0=[1, 0.2])
fitted_ka, fitted_k = best_params

fitted_curve = oral_fit_function(real_time_data, fitted_ka, fitted_k)

plt.plot(t, C, label="Analytical")
plt.plot(t, solution.y[0], linestyle="--", label="Numerical (ODE)")
plt.legend()
plt.xlabel("Time (hours)")
plt.ylabel("Concentration (mg/L)")

plt.figure()
plt.plot(t, solution_oral.y[1], color="green", label="Oral dosing (blood concentration)")
plt.xlabel("Time (hours)")
plt.ylabel("Concentration (mg/L)")
plt.legend()

plt.figure()
plt.plot(times_list, blood_list, color="purple")
plt.xlabel("Time (hours)")
plt.ylabel("Concentration (mg/L)")
plt.title("Blood concentration with a dose every 8 hours")

plt.figure()
plt.plot(t, C, label="Linear elimination")
plt.plot(t, solution_saturable.y[0], label="Saturable (Michaelis-Menten) elimination")
plt.xlabel("Time (hours)")
plt.ylabel("Concentration (mg/L)")
plt.title("Linear vs. saturable elimination")
plt.legend()

plt.figure()
plt.scatter(real_time_data, real_concentration_data, color="black", label="Real measured data")
plt.plot(real_time_data, fitted_curve, color="red", label="Fitted model")
plt.xlabel("Time (hours)")
plt.ylabel("Concentration (mg/L)")
plt.title("Model fit to real caffeine-like concentration data")
plt.legend()

plt.show()

print("Fitted absorption rate (ka):", fitted_ka)
print("Fitted elimination rate (k):", fitted_k)