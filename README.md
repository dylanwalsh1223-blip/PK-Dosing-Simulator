# Pharmacokinetic Dosing Simulator

An interactive web app that models how drug concentration in the bloodstream changes over time, built to explore the intersection of biomedical and chemical engineering.

**Live app:** [https://pk-dosing-simulator.streamlit.app/](https://pk-dosing-simulator.streamlit.app/)

## What it does

- Simulates drug absorption and elimination in the body using systems of ordinary differential equations (ODEs), solved numerically with SciPy
- Validates the numerical solver against an exact analytical solution before trusting it on more complex models
- Models oral (pill) dosing as a two-compartment system: drug moving from the gut into the bloodstream, then being eliminated
- Simulates multi-dose regimens, showing how concentration accumulates with repeated doses over time
- Compares linear elimination against nonlinear, saturable (Michaelis-Menten) elimination kinetics — the same math used to describe enzyme-limited reactions in chemical engineering
- Fits the model to real concentration data using nonlinear least-squares curve fitting, as a validation step connecting the theoretical model to real measurements
- Includes presets for 25 common drugs, each with its own absorption rate, elimination rate, and therapeutic window
- Adjusts elimination rate for reduced kidney/liver function, and supports weight-based (mg/kg) dosing
- Lets the user compare two drugs side-by-side to see how their absorption and elimination profiles differ
- Generates a plain-language explanation of each result — whether a dosing schedule is safe, ineffective, or risks toxicity, and why
- Backed by an automated test suite (pytest) that checks the model against known analytical values

## Why I built this

I'm interested in studying biomedical engineering, and I wanted a project that combined the math I was learning in my differential equations class with something with a real, tangible medical application. Pharmacokinetics stood out because it's a field where chemical engineering concepts — reaction rates, mass transfer, compartment modeling — directly explain something everyone encounters: why medications are dosed the way they are, and why some drugs need to be taken every few hours while others only need one dose a day. Building this let me go from solving equations on paper to seeing, visually, why dosing schedules matter and what happens when a model doesn't match a patient's physiology.

## Tech used

- **Python** — core language
- **NumPy** — numerical operations
- **SciPy** — solving differential equations (`solve_ivp`) and fitting the model to real data (`curve_fit`)
- **Streamlit** — turns the model into an interactive web app
- **Plotly** — interactive charting with shaded therapeutic window ranges
- **pytest** — automated tests verifying the model's numerical accuracy

## How it works, briefly

The core model is a two-compartment ODE system:

    dA/dt = -ka * A
    dC/dt = ka * A - k * C

where `A` is drug remaining in the gut, `C` is drug concentration in the blood, `ka` is the absorption rate, and `k` is the elimination rate. Multiple doses are simulated by solving this system in sequence, adding a new dose to the gut compartment at each dosing interval and carrying the ending state forward into the next interval.

## Data & references

The compartment modeling approach and the Michaelis-Menten (saturable) elimination model used in this project are standard pharmacokinetics concepts, described in:

- Gibaldi, M. and Perrier, D. *Pharmacokinetics*, 2nd ed. Marcel Dekker, New York, 1982.
- NCBI Bookshelf, StatPearls: ["Elimination Half-Life of Drugs"](https://www.ncbi.nlm.nih.gov/books/NBK554498/), National Library of Medicine, National Institutes of Health.

Drug-specific absorption rates, elimination rates, and therapeutic window values used as presets in this app are educational approximations, estimated from published elimination half-life ranges for each drug and converted into rate constants for demonstration purposes. They are not sourced from patient-specific clinical data and should not be used for real dosing decisions.

## Disclaimer

This is an educational simulation built with approximate, illustrative parameters. It is not intended for clinical use, real dosing decisions, or medical advice.