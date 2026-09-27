import numpy as np
from scipy.integrate import solve_ivp

def analytical_solution(t, C0, k):
    return C0 * np.exp(-k * t)

def ode_model(t, C, k):
    return -k * C

def oral_model(t, y, ka, k):
    A, C = y
    dA_dt = -ka * A
    dC_dt = ka * A - k * C
    return [dA_dt, dC_dt]

def test_analytical_solution_at_time_zero():
    result = analytical_solution(0, 100, 0.1)
    assert result == 100

def test_analytical_solution_decreases_over_time():
    early_value = analytical_solution(1, 100, 0.1)
    later_value = analytical_solution(10, 100, 0.1)
    assert later_value < early_value

def test_analytical_solution_known_value():
    result = analytical_solution(1, 100, 0.1)
    expected = 90.48
    assert abs(result - expected) < 0.01

def test_numerical_solver_matches_analytical_solution():
    t_points = np.linspace(0, 24, 100)
    analytical_values = analytical_solution(t_points, 100, 0.1)
    numerical_result = solve_ivp(ode_model, [0, 24], [100], args=(0.1,), t_eval=t_points)
    numerical_values = numerical_result.y[0]
    max_difference = np.max(np.abs(analytical_values - numerical_values))
    assert max_difference < 0.1

def test_oral_dosing_starts_at_zero_blood_concentration():
    solution = solve_ivp(oral_model, [0, 24], [100, 0], args=(0.5, 0.1), t_eval=np.linspace(0, 24, 100))
    first_blood_value = solution.y[1][0]
    assert first_blood_value == 0

def test_oral_dosing_blood_concentration_eventually_rises():
    solution = solve_ivp(oral_model, [0, 24], [100, 0], args=(0.5, 0.1), t_eval=np.linspace(0, 24, 100))
    blood_values = solution.y[1]
    assert max(blood_values) > 0