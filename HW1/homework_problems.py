# homework_problems.py
# รวมฟังก์ชันทั้งหมดสำหรับการบ้าน

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from IPython.display import display, clear_output
import time

# --- ฟังก์ชันสำหรับข้อ 1 ---
def calculate_stationary_distribution():
    T_matrix = np.array([[0.9, 0.1], [0.3, 0.7]])
    eigenvalues, eigenvectors = np.linalg.eig(T_matrix.T)
    idx = np.isclose(eigenvalues, 1).argmax()
    stationary_vector = eigenvectors[:, idx].real
    return stationary_vector / stationary_vector.sum()

# --- ฟังก์ชันสำหรับข้อ 2 ---
def run_simple_particle_filter():
    # Model Functions (Nested inside for encapsulation)
    def transition_model(particles, space_size, std_dev=2.0):
        noise = np.random.randn(len(particles)) * std_dev
        particles += noise
        return np.clip(particles, 0, space_size)

    def calculate_weights(particles, measurement, std_dev=5.0):
        weights = norm.pdf(measurement, loc=particles, scale=std_dev)
        weights += 1e-300
        return weights / np.sum(weights)

    def resample(particles, weights):
        indices = np.random.choice(len(particles), size=len(particles), p=weights)
        return particles[indices]

    # Simulation
    space_size, num_particles, true_position = 100, 1000, 20.0
    particles = np.random.uniform(0, space_size, num_particles)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for step in range(15):
        true_position = transition_model(np.array([true_position]), space_size)[0]
        measurement = true_position + np.random.randn() * 5.0
        
        particles = transition_model(particles, space_size)
        weights = calculate_weights(particles, measurement)
        particles = resample(particles, weights)
        
        ax.clear()
        ax.hist(particles, bins=50, range=(0, space_size), density=True, label='Belief')
        ax.axvline(true_position, color='r', ls='--', label='True Position')
        ax.axvline(measurement, color='g', ls=':', label='Measurement')
        ax.set_title(f'Simple Particle Filter - Step {step+1}')
        ax.legend(loc='upper right')
        ax.set_xlim(0, space_size); ax.set_ylim(0, 0.2)
        display(fig)
        clear_output(wait=True)
        time.sleep(0.5)
    plt.show()

# --- ฟังก์ชันสำหรับข้อ 3 ---
def run_hmm_analysis():
    pi = np.array([0.6, 0.4]); A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.8, 0.2], [0.1, 0.9]]); obs_seq = [1, 1, 1]
    obs_map = ['No Symptoms', 'Symptoms']
    
    num_states, T = A.shape[0], len(obs_seq)
    beliefs = np.zeros((T, num_states)); alpha = np.zeros((T, num_states))
    
    alpha[0, :] = pi * B[:, obs_seq[0]]
    beliefs[0, :] = alpha[0, :] / np.sum(alpha[0, :])
    
    for t in range(1, T):
        alpha[t, :] = (alpha[t-1, :] @ A) * B[:, obs_seq[t]]
        beliefs[t, :] = alpha[t, :] / np.sum(alpha[t, :])
        
    print("Observation Sequence:", [obs_map[o] for o in obs_seq])
    print("-" * 40)
    for t, belief in enumerate(beliefs):
        print(f"Day {t+1} Belief: P(Healthy)={belief[0]:.4f}, P(Sick)={belief[1]:.4f}")

# --- ฟังก์ชันสำหรับข้อ 4 ---
def plot_sensor_model():
    measurement = 25.0
    positions = np.linspace(0, 100, 200)
    likelihoods = norm.pdf(measurement, loc=np.abs(positions - 0), scale=3.0)
    
    plt.figure(figsize=(10, 5))
    plt.plot(positions, likelihoods)
    plt.title(f"Sensor Model Likelihood: P(E={measurement}m | X)")
    plt.xlabel("Robot Position (X)"); plt.ylabel("Likelihood")
    plt.grid(True); plt.show()

# --- ฟังก์ชันสำหรับข้อ 5 ---
def run_kalman_filter():
    dt, steps = 1.0, 50; F = np.array([[1, dt], [0, 1]]); H = np.array([[1, 0]])
    Q = np.eye(2)*0.05; R = np.array([[2.0]])
    
    true_states = np.zeros((steps, 2)); true_states[0, :] = [0, 0.5]
    for t in range(1, steps):
        true_states[t,:] = F @ true_states[t-1,:] + np.random.multivariate_normal([0,0],Q)
    measurements = true_states[:,0] + np.random.randn(steps)*np.sqrt(R[0,0])
    
    x_hat = np.array([0,0]); P = np.eye(2); estimates = np.zeros((steps, 2))
    for t in range(steps):
        x_hat_minus = F @ x_hat; P_minus = F @ P @ F.T + Q
        K = P_minus @ H.T @ np.linalg.inv(H @ P_minus @ H.T + R)
        x_hat = x_hat_minus + K @ (measurements[t] - H @ x_hat_minus)
        P = (np.eye(2) - K @ H) @ P_minus
        estimates[t, :] = x_hat
        
    plt.figure(figsize=(12, 6))
    plt.plot(true_states[:,0],'k-',label='True Position')
    plt.plot(measurements,'r.',label='Measurements')
    plt.plot(estimates[:,0],'g-',label='Kalman Estimate')
    plt.title('Kalman Filter (Linear System)'); plt.legend(); plt.grid(True); plt.show()

def run_nonlinear_particle_filter():
    num_particles, steps, r, omega = 2000, 100, 20.0, 0.1
    thetas = np.linspace(0, omega*steps, steps)
    true_path = r*np.c_[np.cos(thetas), np.sin(thetas)]
    particles = np.random.randn(num_particles, 2)*5 + np.array([r,0])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    for t in range(steps):
        x, y = particles[:, 0], particles[:, 1]
        theta_new = np.arctan2(y, x) + omega + np.random.randn(num_particles)*0.05
        particles[:,0] = r*np.cos(theta_new) + np.random.randn(num_particles)*0.5
        particles[:,1] = r*np.sin(theta_new) + np.random.randn(num_particles)*0.5
        
        measurement = np.linalg.norm(true_path[t]) + np.random.randn()*2.0
        weights = norm.pdf(measurement,loc=np.linalg.norm(particles,axis=1),scale=2.0)
        indices = np.random.choice(num_particles,size=num_particles,p=weights/np.sum(weights))
        particles = particles[indices]
        
        if t % 10 == 0:
            ax.clear()
            ax.plot(true_path[:t+1,0], true_path[:t+1,1], 'k-', label='True Path')
            ax.scatter(particles[:,0], particles[:,1], s=5, alpha=0.4, label='Particles')
            ax.set_title(f'Non-Linear Particle Filter (Step {t})')
            ax.legend(); ax.axis('equal'); ax.grid(True)
            display(fig); clear_output(wait=True); time.sleep(0.1)
    plt.show()