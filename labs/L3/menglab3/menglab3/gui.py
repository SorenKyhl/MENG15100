import numpy as np
import pandas as pd
import plotly.graph_objects as go

def interactive_projection(projection_problem=False, df=None, n=250, random_state=42):
    """
    Build an interactive visualization using a generated 3D dataset.
    - If df is None, calls generate_projection_data(n, random_state).
    - Uses consistent colors across 3D and 2D projections.
    - Fixes subplot domains to prevent layout shifts when switching projections.
    """
    # ----------------------------
    # 1) Generate / load data
    # ----------------------------
    if df is None:
        df = generate_projection_data(projection_problem, n=n, random_state=random_state)

    # Split clusters
    df_A = df[df["cluster"] == "A"]
    df_B = df[df["cluster"] == "B"]

    # Numpy arrays for speed
    x1, y1, z1 = df_A["x"].to_numpy(), df_A["y"].to_numpy(), df_A["z"].to_numpy()
    x2, y2, z2 = df_B["x"].to_numpy(), df_B["y"].to_numpy(), df_B["z"].to_numpy()

    # ----------------------------
    # 2) Visual config
    # ----------------------------
    COLOR_A = "#1f77b4"  # blue
    COLOR_B = "#d62728"  # red

    # Domains (left: 3D scene, right: 2D cartesian)
    SCENE_DOMAIN = dict(x=[0.0, 0.58], y=[0.1, 1.0])  # leave space at bottom for legend
    XAXIS_DOMAIN = [0.64, 1.0]
    YAXIS_DOMAIN = [0.0, 1.0]

    # ----------------------------
    # 3) Build figure
    # ----------------------------
    fig = go.Figure()

    # 3D traces (left)
    fig.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z1, mode="markers",
        marker=dict(size=4, opacity=0.85, color=COLOR_A),
        name="Cluster A (lower Z)", legendgroup="A"
    ))
    fig.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z2, mode="markers",
        marker=dict(size=4, opacity=0.85, color=COLOR_B),
        name="Cluster B (higher Z)", legendgroup="B"
    ))

    # 2D projection traces (right) — keep colors consistent; hide from legend
    proj_xy_a = go.Scatter(x=x1, y=y1, mode="markers",
                           marker=dict(size=6, opacity=0.85, color=COLOR_A),
                           name="A (XY)", legendgroup="A", showlegend=False)
    proj_xy_b = go.Scatter(x=x2, y=y2, mode="markers",
                           marker=dict(size=6, opacity=0.85, color=COLOR_B),
                           name="B (XY)", legendgroup="B", showlegend=False)
    proj_xz_a = go.Scatter(x=x1, y=z1, mode="markers",
                           marker=dict(size=6, opacity=0.85, color=COLOR_A),
                           name="A (XZ)", legendgroup="A", showlegend=False)
    proj_xz_b = go.Scatter(x=x2, y=z2, mode="markers",
                           marker=dict(size=6, opacity=0.85, color=COLOR_B),
                           name="B (XZ)", legendgroup="B", showlegend=False)
    proj_yz_a = go.Scatter(x=y1, y=z1, mode="markers",
                           marker=dict(size=6, opacity=0.85, color=COLOR_A),
                           name="A (YZ)", legendgroup="A", showlegend=False)
    proj_yz_b = go.Scatter(x=y2, y=z2, mode="markers",
                           marker=dict(size=6, opacity=0.85, color=COLOR_B),
                           name="B (YZ)", legendgroup="B", showlegend=False)

    fig.add_traces([proj_xy_a, proj_xy_b, proj_xz_a, proj_xz_b, proj_yz_a, proj_yz_b])

    # ----------------------------
    # 4) Fix domains so subplots never overlap
    # ----------------------------
    fig.update_layout(
        # Left: 3D scene
        scene=dict(
            domain=SCENE_DOMAIN,
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="cube",
        ),
        # Right: 2D cartesian (these are the ONLY cartesian axes)
        xaxis=dict(domain=XAXIS_DOMAIN, anchor="y", title="X"),
        yaxis=dict(domain=YAXIS_DOMAIN, anchor="x", title="Y"),
        margin=dict(l=40, r=20, t=60, b=60),
        title_text="3D stacked clusters (left) with 2D projection (right)",
    )

    # Legend positioned below the 3D scene only
    fig.update_layout(
        legend=dict(
            orientation="h",
            x=0.0,  # aligns with left edge of scene domain
            xanchor="left",
            y=0.08,  # just above the bottom of the figure, under the scene
            yanchor="top",
            traceorder="normal",
            bgcolor="rgba(0,0,0,0)"
        )
    )

    # ----------------------------
    # 5) Initial visibility masks
    # Trace order: [3D_A, 3D_B, XY_A, XY_B, XZ_A, XZ_B, YZ_A, YZ_B]
    # ----------------------------
    visible_xy = [True, True, True, True, False, False, False, False]
    visible_xz = [True, True, False, False, True, True, False, False]
    visible_yz = [True, True, False, False, False, False, True, True]
    for i, vis in enumerate(visible_xy):
        fig.data[i].visible = vis

    # Helper to build button layout args WITHOUT losing domains
    def layout_args(x_title, y_title):
        return {
            # Always re-assert domains + anchors when updating titles
            "xaxis": {"title": {"text": x_title}, "domain": XAXIS_DOMAIN, "anchor": "y"},
            "yaxis": {"title": {"text": y_title}, "domain": YAXIS_DOMAIN, "anchor": "x"},
            # Don't touch 'scene' layout here so 3D panel stays put
        }

    # ----------------------------
    # 6) Buttons: switch projection plane
    # ----------------------------
    buttons = [
        dict(
            label="Project onto XY",
            method="update",
            args=[{"visible": visible_xy}, layout_args("X", "Y")]
        ),
        dict(
            label="Project onto XZ",
            method="update",
            args=[{"visible": visible_xz}, layout_args("X", "Z")]
        ),
        dict(
            label="Project onto YZ",
            method="update",
            args=[{"visible": visible_yz}, layout_args("Y", "Z")]
        ),
    ]
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="down", buttons=buttons,
            x=0.62, y=1.12, xanchor="left", yanchor="top",
            pad={"r": 6, "t": 6}
        )]
    )

    fig.show()


import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, BoundedFloatText, HBox, interactive_output
import ipywidgets as widgets
import matplotlib

def interactive_PCA_rotation(X):
  # Wider layout for the two subplots
  matplotlib.rcParams['figure.figsize'] = [10, 4.8]

  AX_MIN, AX_MAX = -8, 8  # fixed axis limits for the left subplot

  def plot_rotated_axes(theta_deg):
      """
      Rotate a pair of orthonormal axes by theta_deg and show:
        - Left: data scatter + rotated axes, with variance labels (fixed limits)
        - Right: bar chart of variances along the two rotated directions
      """
      theta = np.radians(theta_deg)
      rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta),  np.cos(theta)]])

      # Unit vectors (columns of the rotation matrix)
      unit_vec1 = rotation_matrix[:, 0]
      unit_vec2 = rotation_matrix[:, 1]

      # Project data onto rotated unit vectors
      proj1 = X @ unit_vec1
      proj2 = X @ unit_vec2

      # Variances (population)
      var1 = np.var(proj1)
      var2 = np.var(proj2)
      total_var = var1 + var2
      r1 = var1 / total_var if total_var > 0 else 0.0
      r2 = var2 / total_var if total_var > 0 else 0.0

      # Figure with two subplots
      fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2.2, 1.0]})

      # --- Left: scatter + smaller arrows ---
      ax_scatter.scatter(X[:, 0], X[:, 1], alpha=0.3, edgecolor='k', label='Data')

      origin = np.mean(X, axis=0)
      scale = 2.0  # arrow length

      arrow1 = unit_vec1 * scale
      arrow2 = unit_vec2 * scale
      ax_scatter.quiver(*origin, *arrow1, color='red', angles='xy', scale_units='xy', scale=1, label='PC1')
      ax_scatter.quiver(*origin, *arrow2, color='blue', angles='xy', scale_units='xy', scale=1, label='PC2')

      # Variance labels at arrow tips
      tip1 = origin + arrow1
      tip2 = origin + arrow2
      ax_scatter.text(tip1[0], tip1[1], f"Var(PC1) = {var1:.2f}", color='red', fontsize=10, ha='left', va='bottom')
      ax_scatter.text(tip2[0], tip2[1], f"Var(PC2) = {var2:.2f}", color='blue', fontsize=10, ha='left', va='bottom')

      # Fixed axes
      ax_scatter.set_xlim(AX_MIN, AX_MAX)
      ax_scatter.set_ylim(AX_MIN, AX_MAX)

      # Total variance annotation (axes coords so it stays put)
      ax_scatter.text(0.98, 0.02, f"Total variance: {total_var:.2f}",
                      transform=ax_scatter.transAxes, ha='right', va='bottom',
                      fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

      ax_scatter.set_title(f"Rotated Axes at θ = {theta_deg:.1f}°")
      ax_scatter.set_aspect('equal', adjustable='box')
      ax_scatter.set_xlabel('X1')
      ax_scatter.set_ylabel('X2')
      ax_scatter.grid(True, alpha=0.3)
      ax_scatter.legend()

      # --- Right: variance bar chart ---
      labels = ['PC1', 'PC2']
      bars = ax_bar.bar(labels, [var1, var2])
      ax_bar.set_ylabel('Variance')
      ax_bar.set_title('Variance along rotated axes')

      for rect, v, r in zip(bars, [var1, var2], [r1, r2]):
          ax_bar.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                      f"{v:.2f}\n({r:.0%})", ha='center', va='bottom')

      ax_bar.set_ylim(0, max(var1, var2) * 1.15 if total_var > 0 else 1.0)
      ax_bar.grid(True, axis='y', alpha=0.3)

      plt.tight_layout()
      plt.show()

  # Controls: slider + type-in box (both 0–180°) and linked two-ways in Python
  slider = FloatSlider(value=0, min=0, max=180, step=1, description='θ (deg)', continuous_update=True)
  box = BoundedFloatText(value=0, min=0, max=180, step=1, description='θ input')

  # Two-way Python-side link (more reliable than jslink). Keep a reference to avoid GC.
  ui = HBox([slider, box])
  ui._link = widgets.link((slider, 'value'), (box, 'value'))

  # Render when either control changes (they're linked so either will update `slider.value`)
  out = interactive_output(plot_rotated_axes, {'theta_deg': slider})

  display(ui, out)

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, BoundedFloatText, HBox, interactive_output
import ipywidgets as widgets
import matplotlib

def interactive_PCA_projection(X):
  # Wider layout for two subplots
  matplotlib.rcParams['figure.figsize'] = [10, 4.8]

  AX_MIN, AX_MAX = -8, 8  # fixed axis range

  def plot_rotated_axes(theta_deg):
      """
      Assumes X is a (n,2) NumPy array already defined.
      Left: rotated data with PC1 projection lines (fixed axes -8..8).
      Right: histogram of PC1 projections (x-axis fixed -8..8).
      """
      theta = np.radians(theta_deg)

      # Rotation matrix (counter-clockwise)
      R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]], dtype=float)

      # Rotate data -> "PC" frame at angle theta
      X_rotated = X @ R

      # Projections
      proj_pc1 = X_rotated[:, 0]
      proj_pc2 = X_rotated[:, 1]

      # Variances (for annotation)
      var_pc1 = np.var(proj_pc1)
      var_pc2 = np.var(proj_pc2)

      # --- Figure: left scatter, right PC1 histogram ---
      fig, (ax_scatter, ax_hist) = plt.subplots(
          1, 2, gridspec_kw={'width_ratios': [2.2, 1.0]}
      )

      # Left: rotated scatter + projection lines onto PC1
      ax_scatter.scatter(
          X_rotated[:, 0], X_rotated[:, 1],
          alpha=0.3, edgecolor='k', label='Rotated Data',
          color='skyblue', s=16, linewidths=0.2
      )
      ax_scatter.scatter(
          X_rotated[:, 0], np.zeros_like(X_rotated[:, 0]),
          alpha=0.7, label='Projections onto PC1',
          color='blue', marker='x', s=14
      )
      for x, y in X_rotated:
          ax_scatter.plot([x, x], [0, y], color='blue', alpha=0.07, lw=1)

      # FIXED axes regardless of theta
      ax_scatter.set_xlim(AX_MIN, AX_MAX)
      ax_scatter.set_ylim(AX_MIN, AX_MAX)

      # Variance annotation (bottom-right corner of fixed box)
      ax_scatter.text(
          AX_MAX, AX_MIN, f"Var(PC1) = {var_pc1:.2f}\nVar(PC2) = {var_pc2:.2f}",
          ha='right', va='bottom', fontsize=9,
          bbox=dict(facecolor='white', alpha=0.6)
      )

      ax_scatter.set_title(f"Rotated Data and PC1 Projections at θ = {theta_deg:.1f}°")
      ax_scatter.set_xlabel('PC1')
      ax_scatter.set_ylabel('PC2')
      ax_scatter.set_aspect('equal', adjustable='box')
      ax_scatter.grid(True, alpha=0.3)
      ax_scatter.legend(loc='upper right')

      # Right: histogram of PC1 projections (fixed x range)
      bins = np.linspace(AX_MIN, AX_MAX, 41)  # fixed bin edges for consistency
      ax_hist.hist(proj_pc1, bins=bins, density=True)
      ax_hist.set_xlim(AX_MIN, AX_MAX)
      ax_hist.set_title('Histogram of projections onto PC1')
      ax_hist.set_xlabel('PC1 value')
      ax_hist.set_ylabel('Density')
      ax_hist.grid(True, axis='y', alpha=0.3)

      plt.tight_layout()
      plt.show()

  # --- Controls: slider + type-in box (linked two ways) ---
  slider = FloatSlider(value=0, min=0, max=180, step=1, description='θ (deg)', continuous_update=True)
  box = BoundedFloatText(value=0, min=0, max=180, step=1, description='θ input')

  # Two-way Python-side link; keep a reference to avoid garbage collection
  ui = HBox([slider, box])
  ui._link = widgets.link((slider, 'value'), (box, 'value'))

  # Render output when the (linked) value changes
  out = interactive_output(plot_rotated_axes, {'theta_deg': slider})

  display(ui, out)


def scree_plot(pca, title='Scree Plot'):
    var = pca.explained_variance_ratio_
    pcs = np.arange(1, len(var)+1)
    plt.figure()
    plt.plot(pcs, var, marker='o')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(title)
    plt.show()

def pca_scatter(X_pca, y=None, pcx=1, pcy=2, title='PCA Scatter'):
    i, j = pcx-1, pcy-1
    plt.figure()
    if y is None:
        plt.scatter(X_pca[:, i], X_pca[:, j], alpha=0.7)
    else:
        # If labels are provided, color by label index
        uniq = pd.unique(y)
        for k in uniq:
            mask = (y == k)
            plt.scatter(X_pca[mask, i], X_pca[mask, j], alpha=0.7, label=str(k))
        plt.legend()
    plt.xlabel(f'PC{pcx}')
    plt.ylabel(f'PC{pcy}')
    plt.title(title)
    plt.show()

def elbow_plot(X, k_min=2, k_max=10, random_state=0):
    inertias = []
    ks = list(range(k_min, k_max+1))
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.figure()
    plt.plot(ks, inertias, marker='o')
    plt.xlabel('k')
    plt.ylabel('Inertia (within-cluster sum of squares)')
    plt.title('Elbow Plot')
    plt.show()
    return ks, inertias

def silhouette_scan(X, k_min=2, k_max=10, random_state=0):
    scores = []
    ks = list(range(k_min, k_max+1))
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
    plt.figure()
    plt.plot(ks, scores, marker='o')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores by k')
    plt.show()
    return ks, scores

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import Dropdown, Checkbox, HBox, VBox, interactive_output
import ipywidgets as widgets

def interactive_2d_projection(X_wine):
    """
    Interactive 2D projection of the Wine dataset onto two user-selected features.
    - No class coloring
    - No class means
    - No alpha/size controls
    - Optional standardization (z-score)
    """
    cols = list(X_wine.columns)

    # Widgets
    x_feat = Dropdown(options=cols, value=cols[0], description='X feature')
    y_feat = Dropdown(options=cols, value=cols[1], description='Y feature')
    standardize = Checkbox(value=False, description='Standardize (z-score)')

    def plot(feat_x, feat_y, standardize):
        x = X_wine[feat_x].to_numpy()
        y = X_wine[feat_y].to_numpy()

        if standardize:
            x_std = x.std(ddof=0)
            y_std = y.std(ddof=0)
            x = (x - x.mean()) / (x_std if x_std > 0 else 1.0)
            y = (y - y.mean()) / (y_std if y_std > 0 else 1.0)
            xlabel = f'{feat_x} (z)'
            ylabel = f'{feat_y} (z)'
        else:
            xlabel = feat_x
            ylabel = feat_y

        plt.figure(figsize=(6.5, 5.2))
        plt.scatter(x, y, s=24, c='0.2', alpha=0.85, edgecolor='k', linewidths=0.25)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Wine dataset: two-feature projection (raw)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    out = interactive_output(
        plot,
        {'feat_x': x_feat, 'feat_y': y_feat, 'standardize': standardize}
    )

    controls = VBox([
        HBox([x_feat, y_feat]),
        HBox([standardize]),
    ])

    display(controls, out)


# If needed:  %pip install plotly ipywidgets

import numpy as np
import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from ipywidgets import Checkbox, VBox, interactive_output

def PCA_projection_3D(
    X_pca,
    y=None,
    pcs=(1, 2, 3),
    evr=None,
    title="PCA: 3D Scatter",
    height=600,
    width=800,
    marker_size=4,
    opacity=0.85,
):
    """
    Interactive toggle to color by labels or show a single-color plot.

    - When the box is checked and `y` is provided: colors = Matplotlib defaults
      [blue, orange, green], legend shown.
    - When unchecked (or y is None): single-color (Matplotlib blue), no legend.
    """
    i, j, k = (p - 1 for p in pcs)
    MPL_3 = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Matplotlib default first 3 colors

    # Axis labels (optionally include explained variance %)
    def axis_label(idx0, name):
        if evr is not None and idx0 < len(evr):
            return f"{name} ({evr[idx0]*100:.1f}%)"
        return name

    x_name, y_name, z_name = f"PC{pcs[0]}", f"PC{pcs[1]}", f"PC{pcs[2]}"
    labels = {
        x_name: axis_label(i, x_name),
        y_name: axis_label(j, y_name),
        z_name: axis_label(k, z_name),
        "label": "Class",
    }

    # Checkbox to toggle coloring by labels
    toggle = Checkbox(
        value=(y is not None),
        description="Color by labels",
        disabled=(y is None),  # if no labels provided, disable toggle
    )

    def _render(color_by_labels):
        df = pd.DataFrame({
            x_name: X_pca[:, i],
            y_name: X_pca[:, j],
            z_name: X_pca[:, k],
        })

        color_kwargs = {}
        showlegend = False

        if color_by_labels and (y is not None):
            df["label"] = pd.Series(y).astype(str)
            unique_labels = sorted(df["label"].unique(), key=lambda s: (len(s), s))
            color_kwargs.update(
                color="label",
                category_orders={"label": unique_labels},
                color_discrete_sequence=MPL_3[:len(unique_labels)],
            )
            showlegend = True

        fig = px.scatter_3d(
            df,
            x=x_name, y=y_name, z=z_name,
            opacity=opacity,
            labels=labels,
            title=title,
            height=height, width=width,
            **color_kwargs
        )

        # Style markers
        fig.update_traces(marker=dict(size=marker_size, line=dict(width=0)))

        # If not coloring by labels, use single Matplotlib blue and hide legend
        if not (color_by_labels and (y is not None)):
            fig.update_traces(marker=dict(color=MPL_3[0]))
            showlegend = False

        fig.update_layout(showlegend=showlegend, legend_title_text="Class" if showlegend else None)
        fig.show()

    out = interactive_output(_render, {"color_by_labels": toggle})
    display(VBox([toggle, out]))

# Example usage after PCA:
# pca_scatter_3d_plotly_toggle(X_pca, y=y_wine, pcs=(1,2,3), evr=pca.explained_variance_ratio_)

# Step 1: Setup
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import ipywidgets as widgets

def interactive_k_means(X, y_true):
  # Step 3: K-means with history tracking (no trails)
  def kmeans_with_history(X, k, seed=0, max_iter=20):
      rng = np.random.default_rng(seed)
      centroids = X[rng.choice(X.shape[0], size=k, replace=False)]
      history = []

      for _ in range(max_iter):
          # E-step: Assign clusters
          distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
          labels = np.argmin(distances, axis=1)

          # Record state
          history.append((centroids.copy(), labels.copy()))

          # M-step: Update centroids
          new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

          # Stop if converged
          if np.allclose(centroids, new_centroids):
              break

          centroids = new_centroids

      return history

  # Step 4: Animation function with WCSS annotation
  def animate_kmeans(k, seed):
      history = kmeans_with_history(X, k, seed)
      fig, ax = plt.subplots(figsize=(6, 6))
      colors = plt.cm.tab10.colors

      def update(i):
          ax.clear()
          centroids, labels = history[i]

          # Plot each cluster and its centroid
          for cluster_id in range(k):
              points = X[labels == cluster_id]
              ax.scatter(points[:, 0], points[:, 1], s=30,
                        color=colors[cluster_id % 10],
                        label=f"Cluster {cluster_id}")
              ax.scatter(*centroids[cluster_id], marker='X', color='black', s=200, edgecolor='white')

          # Compute WCSS for current clustering
          wcss = np.sum([
              np.sum((X[labels == cluster_id] - centroids[cluster_id]) ** 2)
              for cluster_id in range(k)
          ])

          ax.set_title(f"K-Means Iteration {i + 1}")
          ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
          ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
          ax.legend(loc='best')

          # Add WCSS text in bottom-right
          ax.text(0.95, 0.02, f"WCSS = {wcss:.2f}",
                  transform=ax.transAxes,
                  ha='right', va='bottom',
                  fontsize=10, color='darkred',
                  bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

      ani = FuncAnimation(fig, update, frames=len(history), interval=1000, repeat=True)
      plt.close()
      display(HTML(ani.to_jshtml()))

  # Step 5: Interactive interface using value input boxes
  k_input = widgets.BoundedIntText(
      value=3,
      min=1,
      max=10,
      step=1,
      description='Number of clusters, k =',
      style={'description_width': 'initial'}
  )

  seed_input = widgets.IntText(
      value=0,
      description='Random number seed =',
      style={'description_width': 'initial'}
  )

  ui = widgets.VBox([k_input, seed_input])
  out = widgets.interactive_output(animate_kmeans, {'k': k_input, 'seed': seed_input})
  display(ui, out)

