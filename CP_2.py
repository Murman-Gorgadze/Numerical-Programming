

import numpy as np
import matplotlib.pyplot as plt
import time


# ==========================================
# 1. MODEL DEFINITION: FitzHugh-Nagumo Model
# ==========================================

class FitzHughNagumo:
    def __init__(self, I_ext=0.5, a=0.7, b=0.8, tau=12.5):
        """
        Initializes the model parameters.
        V: Membrane Potential (Voltage)
        W: Recovery Variable
        I_ext: External Stimulus Current
        """
        self.I_ext = I_ext
        self.a = a
        self.b = b
        self.tau = tau

    def f(self, t, y):
        """
        The System of ODEs: dy/dt = f(t, y)
        y = [V, W]
        """
        V, W = y

        # dV/dt = V - V^3/3 - W + I
        dV_dt = V - (V ** 3) / 3 - W + self.I_ext

        # dW/dt = (V + a - bW) / tau
        dW_dt = (V + self.a - self.b * W) / self.tau

        return np.array([dV_dt, dW_dt])

    def jacobian(self, y):
        """
        Analytic Jacobian Matrix J_f of the system function f(t,y).
        Required for Newton's Method.
        J_f = [[ d(dV)/dV , d(dV)/dW ],
               [ d(dW)/dV , d(dW)/dW ]]
        """
        V, W = y

        # Partial derivatives
        df1_dV = 1 - V ** 2
        df1_dW = -1.0

        df2_dV = 1.0 / self.tau
        df2_dW = -self.b / self.tau

        return np.array([[df1_dV, df1_dW],
                         [df2_dV, df2_dW]])


# ==========================================
# 2. SOLVER IMPLEMENTATIONS (Implicit Euler)
# ==========================================

def solve_implicit_euler_fixed_point(model, y0, t_span, dt, tol=1e-6, max_iter=100):
    """
    Solves the system using Implicit Euler with Fixed Point Iteration.
    y_{n+1} = y_n + dt * f(t_{n+1}, y_{n+1})
    """
    t_start = t_span[0]
    t_end = t_span[1]
    num_steps = int((t_end - t_start) / dt)

    t_values = np.linspace(t_start, t_end, num_steps + 1)
    y_values = np.zeros((num_steps + 1, len(y0)))
    y_values[0] = y0

    total_iterations = 0

    start_time = time.time()

    for i in range(num_steps):
        t_next = t_values[i + 1]
        y_current = y_values[i]

        # Initial Guess for y_{n+1} (using Explicit Euler guess)
        y_next = y_current + dt * model.f(t_values[i], y_current)

        # Fixed Point Iteration Loop
        for k in range(max_iter):
            # G(y) = y_n + dt * f(t_{n+1}, y)
            y_next_new = y_current + dt * model.f(t_next, y_next)

            # Check convergence
            error = np.linalg.norm(y_next_new - y_next)
            y_next = y_next_new

            if error < tol:
                total_iterations += (k + 1)
                break
        else:
            print(f"Warning: Fixed Point did not converge at step {i}")

        y_values[i + 1] = y_next

    end_time = time.time()
    avg_iter = total_iterations / num_steps
    return t_values, y_values, end_time - start_time, avg_iter


def solve_implicit_euler_newton(model, y0, t_span, dt, tol=1e-6, max_iter=50):
    """
    Solves the system using Implicit Euler with Newton's Method.
    We solve for root of: G(y_{n+1}) = y_{n+1} - y_n - dt*f(t_{n+1}, y_{n+1}) = 0
    Jacobian of G: J_G = I - dt * J_f
    """
    t_start = t_span[0]
    t_end = t_span[1]
    num_steps = int((t_end - t_start) / dt)

    t_values = np.linspace(t_start, t_end, num_steps + 1)
    y_values = np.zeros((num_steps + 1, len(y0)))
    y_values[0] = y0

    total_iterations = 0
    identity = np.eye(len(y0))

    start_time = time.time()

    for i in range(num_steps):
        t_next = t_values[i + 1]
        y_current = y_values[i]

        # Initial Guess (Explicit Euler)
        y_next = y_current + dt * model.f(t_values[i], y_current)

        # Newton Iteration Loop
        for k in range(max_iter):
            # Calculate Residual: R = y_{n+1} - y_n - dt*f(y_{n+1})
            f_val = model.f(t_next, y_next)
            residual = y_next - y_current - dt * f_val

            # Calculate Jacobian for Newton Step: J_G = I - dt * J_f
            J_f = model.jacobian(y_next)
            J_G = identity - dt * J_f

            # Solve Linear System: J_G * delta = -Residual
            delta = np.linalg.solve(J_G, -residual)

            y_next = y_next + delta

            # Check convergence
            if np.linalg.norm(delta) < tol:
                total_iterations += (k + 1)
                break
        else:
            print(f"Warning: Newton method did not converge at step {i}")

        y_values[i + 1] = y_next

    end_time = time.time()
    avg_iter = total_iterations / num_steps
    return t_values, y_values, end_time - start_time, avg_iter


# ==========================================
# 3. RUN SIMULATION AND VISUALIZE
# ==========================================

if __name__ == "__main__":
    # --- Configuration ---
    # Parameters that create spiking behavior
    model = FitzHughNagumo(I_ext=0.5, a=0.7, b=1.0, tau=12.5)

    # Initial Condition [V, W]
    y0 = np.array([-1.0, 1.0])


    # Time Span
    t_span = (0, 100)

    # CHANGE 1: Increase time step slightly to make it harder
    dt = 0.2

    print(f"--- Running Simulation (dt={dt}) ---")

    # CHANGE 2: Make tolerance extremely strict (1e-12 instead of 1e-6)
    # --- Run Fixed Point Method ---
    t_fp, y_fp, time_fp, iter_fp = solve_implicit_euler_fixed_point(model, y0, t_span, dt, tol=1e-11, max_iter=200)

    # --- Run Newton Method ---
    t_nw, y_nw, time_nw, iter_nw = solve_implicit_euler_newton(model, y0, t_span, dt, tol=1e-11, max_iter=50)

    # --- Visualization ---
    plt.figure(figsize=(12, 10))

    # Plot 1: Voltage (V) over time
    plt.subplot(2, 2, 1)
    plt.plot(t_fp, y_fp[:, 0], 'b-', label='Fixed Point', linewidth=2, alpha=0.7)
    plt.plot(t_nw, y_nw[:, 0], 'r--', label='Newton', linewidth=2, alpha=0.7)
    plt.title("Membrane Potential (V) vs Time")
    plt.xlabel("Time")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.grid(True)

    # Plot 2: Recovery (W) over time
    plt.subplot(2, 2, 2)
    plt.plot(t_fp, y_fp[:, 1], 'b-', label='Fixed Point', alpha=0.7)
    plt.plot(t_nw, y_nw[:, 1], 'r--', label='Newton', alpha=0.7)
    plt.title("Recovery Variable (W) vs Time")
    plt.xlabel("Time")
    plt.ylabel("Recovery (W)")
    plt.grid(True)

    # Plot 3: Phase Portrait (V vs W) - Classic Neuroscience plot
    plt.subplot(2, 2, 3)
    plt.plot(y_nw[:, 0], y_nw[:, 1], 'g-', linewidth=1.5)
    plt.title("Phase Portrait (Limit Cycle)")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Recovery (W)")
    plt.grid(True)

    # Plot 4: Comparison Table (Visual)
    plt.subplot(2, 2, 4)
    plt.axis('off')
    table_data = [
        ["Method", "Exec Time (s)", "Avg Iterations"],
        ["Fixed Point", f"{time_fp:.4f}", f"{iter_fp:.2f}"],
        ["Newton", f"{time_nw:.4f}", f"{iter_nw:.2f}"]
    ]
    table = plt.table(cellText=table_data, loc='center', cellLoc='center')
    table.scale(1, 2)
    table.set_fontsize(14)
    plt.title("Performance Comparison")

    plt.tight_layout()
    plt.show()

    # --- Console Conclusion ---
    print("\n--- Conclusion ---")
    if time_nw > time_fp:
        print("Observation: Fixed Point was faster per step (no matrix inversion).")
    else:
        print("Observation: Newton was faster (fewer iterations required).")

    print(
        "Newton usually requires fewer iterations (Quadratic convergence) compared to Fixed Point (Linear convergence).")
    print("However, Newton is more computationally expensive per iteration due to solving the linear system.")