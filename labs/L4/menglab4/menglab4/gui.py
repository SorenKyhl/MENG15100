def interactive_activation_functions():
  import numpy as np
  import matplotlib.pyplot as plt
  from ipywidgets import interact, FloatSlider, Dropdown, VBox, HBox, Output
  import ipywidgets as widgets

  def sigmoid(z):
      """Sigmoid activation function"""
      return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to avoid overflow

  def relu(z):
      """ReLU activation function"""
      return np.maximum(0, z)

  def create_activation_explorer():
      """Create interactive widget for exploring activation functions"""
      
      # Create widgets
      activation_dropdown = Dropdown(
          options=['sigmoid', 'relu'],
          value='sigmoid',
          description='Activation:',
          style={'description_width': '100px'}
      )
      
      w_slider = FloatSlider(
          value=1.0,
          min=-3.0,
          max=3.0,
          step=0.1,
          description='w (slope):',
          style={'description_width': '100px'},
          continuous_update=True
      )
      
      b_slider = FloatSlider(
          value=0.0,
          min=-5.0,
          max=5.0,
          step=0.1,
          description='b (shift):',
          style={'description_width': '100px'},
          continuous_update=True
      )
      
      # Output widget for plot
      output = Output()
      
      def update_plot(activation, w, b):
          """Update plot based on widget values"""
          # Create x values
          x = np.linspace(-10, 10, 500).reshape(-1, 1)
          
          # Compute linear transformation
          z = w * x + b
          
          # Apply activation function
          if activation == 'sigmoid':
              y = sigmoid(z)
              activation_func = sigmoid
              y_label = 'σ(w·x + b)'
          else:  # relu
              y = relu(z)
              activation_func = relu
              y_label = 'ReLU(w·x + b)'
          
          # Clear and create new plot
          with output:
              output.clear_output(wait=True)
              
              fig, axes = plt.subplots(1, 3, figsize=(15, 4))
              
              # Plot 1: Linear transformation (before activation)
              ax1 = axes[0]
              ax1.plot(x, z, 'b-', linewidth=2)
              ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
              ax1.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
              ax1.grid(True, alpha=0.3)
              ax1.set_xlabel('x', fontsize=12)
              ax1.set_ylabel('z = w·x + b', fontsize=12)
              ax1.set_title('Step 1: Linear Transformation', fontsize=13, fontweight='bold')
              ax1.set_ylim([-10, 10])
              
              # Add annotation for the kink point (where z=0)
              if w != 0:
                  x_kink = -b / w
                  if -10 <= x_kink <= 10:
                      ax1.plot(x_kink, 0, 'ro', markersize=10, label=f'z=0 at x={x_kink:.2f}')
                      ax1.legend()
              
              # Plot 2: Activation function
              ax2 = axes[1]
              z_range = np.linspace(-10, 10, 500)
              y_range = activation_func(z_range)
              ax2.plot(z_range, y_range, 'g-', linewidth=2)
              ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
              ax2.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
              ax2.grid(True, alpha=0.3)
              ax2.set_xlabel('z', fontsize=12)
              ax2.set_ylabel(f'{activation}(z)', fontsize=12)
              ax2.set_title(f'Step 2: {activation.capitalize()} Activation', fontsize=13, fontweight='bold')
              
              if activation == 'sigmoid':
                  ax2.set_ylim([-0.1, 1.1])
                  # Mark important points
                  ax2.plot(0, 0.5, 'ro', markersize=8, label='Midpoint (0, 0.5)')
                  ax2.legend()
              else:
                  ax2.set_ylim([-1, 10])
                  # Mark the kink
                  ax2.plot(0, 0, 'ro', markersize=8, label='Kink at (0, 0)')
                  ax2.legend()
              
              # Plot 3: Final output
              ax3 = axes[2]
              ax3.plot(x, y, 'r-', linewidth=3)
              ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
              ax3.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
              ax3.grid(True, alpha=0.3)
              ax3.set_xlabel('x', fontsize=12)
              ax3.set_ylabel(y_label, fontsize=12)
              ax3.set_title('Step 3: Final Output', fontsize=13, fontweight='bold')
              
              if activation == 'sigmoid':
                  ax3.set_ylim([-0.1, 1.1])
              else:
                  ax3.set_ylim([-1, 10])
              
              plt.tight_layout()
              plt.show()
      
      # Create layout
      controls = VBox([
          activation_dropdown,
          w_slider,
          b_slider
      ])
      
      # Connect widgets to update function
      interactive_plot = interact(
          update_plot,
          activation=activation_dropdown,
          w=w_slider,
          b=b_slider
      )
      
      # Display output
      display(output)
      
      # Initial plot
      update_plot('sigmoid', 1.0, 0.0)

  # Create the widget
  create_activation_explorer()
