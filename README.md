# HESS-ML

## Machine Learning Based Hybrid Energy Storage Management for Electric Vehicles

HESS-ML is an educational machine-learning project that simulates intelligent energy management for an Electric Vehicle (EV) using a Hybrid Energy Storage System (HESS).

The system combines:

- Battery
- Supercapacitor
- Machine Learning
- Rule-based control
- Simulation
- Interactive Streamlit dashboard

The main objective is to learn an energy-management policy that distributes vehicle power demand between the battery and supercapacitor while considering storage state and system constraints.

---

# Project Overview

Electric vehicles commonly use batteries as their primary energy source.

However, high-power demand can cause large current flow through the battery, resulting in higher estimated resistive losses and increased battery-side power stress.

A Hybrid Energy Storage System combines:

### Battery

The battery provides relatively sustained energy.

### Supercapacitor

The supercapacitor is suitable for handling short-term high-power demand.

The HESS-ML project uses machine learning to learn how much power should be supplied by the battery, while the remaining demand is handled by the supercapacitor.

---

# System Architecture

```text
                    Driving Cycle
                         |
                         v
                Power Demand Data
                         |
                         v
              Random Forest Predictor
                         |
                         v
                Vehicle Power Demand
                         |
                         v
              Teacher Control Policy
                         |
                         v
                  Policy Dataset
                         |
                         v
              Random Forest Student
                         |
                         v
              Battery Power Allocation
                         |
                         v
        +-------------------------------+
        |                               |
        v                               v
     Battery                     Supercapacitor
        |                               |
        +---------------+---------------+
                        |
                        v
                 HESS Simulation
                        |
                        v
               Controller Evaluation