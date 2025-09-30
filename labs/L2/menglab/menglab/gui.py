import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs import FigureWidget
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Output
from IPython.display import display, clear_output, HTML

def data_explorer(data):
    """
    Build and display an interactive mini-app for exploring a molecular dataset.

    This tool shows:
      1) A dropdown to choose a molecule by name.
      2) A left panel with an interactive 3D view of the selected molecule and its key
         properties (name, molecular weight, boiling point).
      3) A right panel with a clickable scatter plot of **Boiling Point (K)** vs.
         **Molecular Weight (MW)**. Clicking a point selects that molecule in the
         dropdown and updates the 3D viewer.

    Designed for beginners: you don't need prior AI/ML knowledge—just run this function
    in a Jupyter notebook and use your mouse to rotate/zoom the 3D molecule and click
    points on the plot.

    Parameters
    ----------
    data : pandas.DataFrame
        A table containing at least these columns:
          - 'name'  : str, the molecule’s English name
          - 'smiles': str, a SMILES string (text that encodes molecular structure)
          - 'MW'    : float, molecular weight (grams per mole, g/mol)
          - 'bp_k'  : float, boiling point (Kelvin)
        Rows with missing values in any of these columns are dropped internally.

    Behavior
    --------
    - The dropdown (top-left) selects a molecule by name.
    - The left panel renders a 3D model from the SMILES string
      (atoms as colored spheres, bonds as sticks; a faint surface shows overall size/shape).
    - The right panel is a scatter plot (MW on x-axis, Boiling Point in Kelvin on y-axis).
      Clicking a point selects that molecule and synchronizes the dropdown and 3D view.
    - The currently selected molecule is highlighted with a larger, distinct marker.

    Returns
    -------
    None
        The function displays interactive widgets inline (intended for Jupyter/IPython).
        Nothing is returned.

    Notes
    -----
    - Requires Jupyter/IPython to display interactive widgets.
    - Uses Plotly for the scatter plot and py3Dmol (via RDKit) for the 3D molecular view.
    - The helper function `molecule_3d_html_from_smiles` performs SMILES → 3D conversion
      and returns HTML for the left panel viewer.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "name": ["Ethanol", "Acetone"],
    ...     "smiles": ["CCO", "CC(=O)C"],
    ...     "MW": [46.07, 58.08],
    ...     "bp_k": [351.5, 329.4],
    ... })
    >>> data_explorer(df)  # displays the dropdown, 3D viewer, and scatter plot
    """

    # -------- Validate DatFrame columns --------
    required = {"name", "smiles", "MW", "bp_k"}
    if not required.issubset(set(data.columns)):
        missing = sorted(required - set(data.columns))
        raise ValueError(f"DataFrame is missing columns: {missing}")

    df = data.dropna(subset=["name", "smiles", "MW", "bp_k"]).copy()
    df = df.sort_values("name")
    names = df["name"].unique().tolist()
    if not names:
        raise ValueError("No valid rows after dropping NaNs.")

    # -------- Widgets / Outputs --------
    w_name = widgets.Dropdown(options=names, value=names[0], description="Molecule:")
    left_out = Output()   # py3Dmol + info

    # -------- Build FigureWidget (clickable) --------
    scatter_all = go.Scatter(
        x=df["MW"].values,
        y=df["bp_k"].values,
        mode="markers",
        name="All molecules",
        marker=dict(size=8, opacity=0.75),
        text=df["name"].values,               # for hover
        customdata=df["name"].values,         # for click callback
        hovertemplate="Name: %{text}<br>MW: %{x:.3f}<br>Boiling Point (K): %{y:.3f}<extra></extra>"
    )

    sel_row = df.loc[df["name"] == w_name.value].iloc[0]
    scatter_sel = go.Scatter(
        x=[float(sel_row["MW"])],
        y=[float(sel_row["bp_k"])],
        mode="markers",
        name=f"Selected: {w_name.value}",
        marker=dict(size=14, color="crimson", line=dict(width=2, color="black")),
        text=[w_name.value],
        hovertemplate="<b>Selected</b><br>Name: %{text}<br>MW: %{x:.3f}<br>Boiling Point (K): %{y:.3f}<extra></extra>"
    )

    figw = FigureWidget(data=[scatter_all, scatter_sel])

    # <<< IMPORTANT: assign a full Layout instead of update_layout >>>
    figw.layout = go.Layout(
        template="plotly_white",
        width=720,
        height=520,
        title=dict(text="Boiling Point vs Molecular Weight (Click to select)", x=0.5, xanchor="center"),
        xaxis=dict(title="Molecular Weight (MW)"),
        yaxis=dict(title="Boiling Point (K)"),
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=1.05, yanchor="bottom"  # centered legend above plot
        ),
        margin=dict(t=90, r=20, b=60, l=70)
    )

    # -------- Helpers to update panels --------
    _internal_update_flag = {"busy": False}  # avoid feedback loops

    def update_left_panel(name: str):
        with left_out:
            clear_output(wait=True)
            row = df.loc[df["name"] == name].iloc[0]
            mol_html = molecule_3d_html_from_smiles(
                row["smiles"], style="ballstick", surface=True, width=500, height=400
            )
            info_html = f"""
            <div style="margin-top:8px;font-family:system-ui,Segoe UI,Helvetica,Arial;">
              <b>Name:</b> {name}<br/>
              <b>Molecular Weight (MW):</b> {float(row['MW']):.1f}<br/>
              <b>Boiling Point (K):</b> {float(row['bp_k']):.1f}
            </div>
            """
            display(HTML(mol_html + info_html))

    def update_selected_trace(name: str):
        row = df.loc[df["name"] == name].iloc[0]
        with figw.batch_update():
            figw.data[1].x = [float(row["MW"])]
            figw.data[1].y = [float(row["bp_k"])]
            figw.data[1].name = f"Selected: {name}"
            figw.data[1].text = [name]

    # -------- Event handlers --------
    def on_dropdown_change(change):
        if change["name"] == "value" and not _internal_update_flag["busy"]:
            name = change["new"]
            update_selected_trace(name)
            update_left_panel(name)

    w_name.observe(on_dropdown_change, names="value")

    def handle_click(trace, points, state):
        if points.point_inds:
            idx = points.point_inds[0]
            clicked_name = trace.customdata[idx]
            _internal_update_flag["busy"] = True
            try:
                w_name.value = clicked_name
            finally:
                _internal_update_flag["busy"] = False
            update_selected_trace(clicked_name)
            update_left_panel(clicked_name)

    figw.data[0].on_click(handle_click)

    # -------- Initial render --------
    update_left_panel(w_name.value)
    display(VBox([w_name, HBox([left_out, figw])]))


# ---------- Helper Function: Plot interactive 3D molecular Viewer ----------
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol

def molecule_3d_html_from_smiles(
      smiles: str,
      style: str = "ballstick",
      surface: bool = True,
      width: int = 520,
      height: int = 400,
      optimize: bool = True,
      add_h: bool = True,
      conf_id: int = -1,
      bg: str = "0xFFFFFF"
    ) -> str:
    """
    helper for `data_explorer` function
    Convert a SMILES string into an interactive 3D molecular viewer (HTML).

    This function uses **RDKit** to generate a 3D structure from a SMILES string,
    optionally optimizes the geometry, and then renders it with **py3Dmol**.
    The result is an HTML snippet that can be displayed inline in Jupyter
    notebooks to explore molecules in 3D.

    Parameters
    ----------
    smiles : str
        A SMILES string representing the molecule (text encoding of atoms
        and bonds).
    style : str, default="ballstick"
        Rendering style for atoms and bonds. Options include:
        - "stick": sticks for bonds only
        - "line": thin lines for bonds
        - "ballstick": sticks plus small spheres for atoms
        - "sphere": atoms as full spheres
        - "cartoon": ribbon/cartoon style (mainly for proteins)
    surface : bool, default=True
        If True, draws a semi-transparent van der Waals (VDW) surface
        around the molecule.
    width : int, default=520
        Width of the viewer in pixels.
    height : int, default=400
        Height of the viewer in pixels.
    optimize : bool, default=True
        If True, run a quick force-field optimization (UFF) after embedding
        the molecule to improve geometry.
    add_h : bool, default=True
        If True, add implicit hydrogens before generating 3D coordinates.
    conf_id : int, default=-1
        Conformer ID to render (useful if multiple conformers are present).
        -1 means "use the default conformer".
    bg : str, default="0xFFFFFF"
        Background color in hex (e.g. "0xFFFFFF" = white, "0x000000" = black).

    Returns
    -------
    str
        An HTML string containing the interactive 3D viewer.
        Can be displayed inline in Jupyter with `IPython.display.HTML`.

    Raises
    ------
    None directly, but returns HTML error messages in two cases:
    - Invalid SMILES string
    - Failure to embed a 3D structure

    Notes
    -----
    - Uses RDKit’s ETKDGv3 algorithm for 3D embedding.
    - The random seed is fixed for reproducibility.
    - If optimization fails, the function falls back to the embedded geometry.
    - Intended for use inside Jupyter/IPython environments.

    Example
    -------
    >>> from IPython.display import HTML
    >>> html = molecule_3d_html_from_smiles("CCO", style="ballstick")
    >>> display(HTML(html))  # Shows ethanol in 3D
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "<b>Invalid SMILES.</b>"
    if add_h:
        mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    if AllChem.EmbedMolecule(mol, params) != 0:
        return "<b>3D embedding failed for this SMILES.</b>"

    if optimize:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass

    mb = Chem.MolToMolBlock(mol, confId=conf_id)

    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor(bg)
    view.addModel(mb, 'sdf')

    style_map = {
        "stick": {"stick": {}},
        "line": {"line": {}},
        "ballstick": {"stick": {}, "sphere": {"scale": 0.25}},
        "sphere": {"sphere": {}},
        "cartoon": {"cartoon": {}}
    }
    view.setStyle(style_map.get(style, {"stick": {}}))
    if surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.4, "color": "white"})
    view.zoomTo()
    return view._make_html()

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def interactive_loss_landscape(X_train, y_train):
    """
    Visualize the loss landscape of linear regression as an interactive plot.

    This function computes the **Root Mean Squared Error (RMSE) loss** across a
    grid of slope ($w$) and intercept ($b$) values, then displays two views:

    1. **3D surface plot** of the loss landscape $L(w, b)$
    2. **2D heatmap** of the same landscape, with colors representing loss values

    The visualization helps illustrate how different choices of $w$ and $b$
    affect the model’s performance, and shows the “valley” where the optimal
    parameters lie.

    Steps
    -----
    1. Fit a simple line to the training data (`X_train`, `y_train`) to
       estimate a reasonable center point for the $(w, b)$ grid.
    2. Generate a meshgrid of $w$ and $b$ values around this center.
    3. Compute the RMSE loss for each pair $(w, b)$ using `loss_function`.
    4. Create a Plotly figure with two subplots:
       - Left: 3D surface of $L(w, b)$
       - Right: Heatmap of $L(w, b)$

    Returns
    -------
    None
        Displays an interactive Plotly figure with both the 3D surface and heatmap.

    Notes
    -----
    - Interactive controls: click, drag, and zoom in the 3D surface for exploration.

    Example
    -------
    >>> interactive_loss_landscape(X_train, y_train)
    # Opens an interactive 3D + heatmap visualization of the loss surface
    """
    def loss_function(y, y_pred):
        return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(y_pred))**2)))
    
    # --- 1. Compute grid for (w, b) ---
    w_ls, b_ls = np.polyfit(np.asarray(X_train).ravel(), np.asarray(y_train).ravel(), 1)

    w_span = 2.0 * max(1.0, abs(w_ls))
    b_span = 2.0 * max(1.0, abs(b_ls) if not np.isclose(b_ls, 0.0) else np.std(y_train))

    W = np.linspace(w_ls - w_span, w_ls + w_span, 100)
    B = np.linspace(b_ls - b_span, b_ls + b_span, 100)
    WW, BB = np.meshgrid(W, B)

    x1d = np.asarray(X_train).ravel()
    y1d = np.asarray(y_train).ravel()

    Z = np.zeros_like(WW, dtype=float)
    for i in range(WW.shape[0]):
        for j in range(WW.shape[1]):
            yhat = WW[i, j] * x1d + BB[i, j]
            Z[i, j] = loss_function(y1d, yhat)

    # --- 2. Make a subplot: 3D surface + heatmap ---
    fig = make_subplots(rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'heatmap'}]],
        subplot_titles=("3D Loss Landscape", "Heatmap of Loss")
    )

    # 3D surface
    fig.add_trace(go.Surface(z=Z, x=WW, y=BB, showscale=False),row=1, col=1)

    # Heatmap
    fig.add_trace(go.Heatmap(z=Z, x=W, y=B, colorbar=dict(title="Loss")),row=1, col=2)

    # --- 3. Layout ---
    fig.update_layout(
        height=500, width=1000,
        scene=dict(
            xaxis_title="w (slope)",
            yaxis_title="b (intercept)",
            zaxis_title="Loss L(w,b)"
        ),
        xaxis2=dict(title="w (slope)"),
        yaxis2=dict(title="b (intercept)")
    )

    fig.show()


# --- Interactive Linear Regression Trainer: adjust w and b with sliders ---
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, HBox, VBox, HTML, interactive_output
from sklearn.linear_model import LinearRegression

def manually_train_linear_regression(X_train, y_train):

    # helper functions
    def loss_function(y, y_pred):
        return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(y_pred))**2)))
        
    def model(x, w, b):
        """Linear Regression Model"""
        return w*x + b
        
    # Set up sliders for w and b
    w_slider = FloatSlider(description="w (slope)", value=1.0, min=-5.0, max=5.0, step=0.05, readout_format=".2f")
    b_slider = FloatSlider(description="b (intercept)", value=1.0, min=-100.0, max=200.0, step=0.5, readout_format=".1f")
    status_label = HTML("<b>RMSE:</b> (move sliders)")
    
    linreg = LinearRegression()
    linreg.fit(X_train.reshape(-1,1), y_train)
    y_best = linreg.predict(X_train.reshape(-1,1))
    min_rmse = loss_function(y_train, y_best)
    
    # Plotting function that updates with the sliders
    def update_plot(w, b):
      y_pred = model(X_train.flatten(), w, b)
      current_rmse = loss_function(y_train, y_pred)
    
      plt.figure(figsize=(6.5, 4.5))
      plt.scatter(X_train, y_train, color='k',label="Data")
      xs = np.linspace(X_train.min(), X_train.max(), 200)
      ys = model(xs, w, b)
    
      plt.plot(xs, ys, color='r', label="Your model")
      plt.xlabel("Molecular weight (x)")
      plt.ylabel("Boiling point (y)")
      plt.title("Adjust w and b to Fit the Data")
      plt.legend()
      plt.grid(True)
      plt.show()
    
      # Update status
      if current_rmse <= 1.2 * min_rmse:  # within 20% of best possible
          status_label.value = f"<b>RMSE:</b> {current_rmse:.3f} ✅ SUCCESS!"
      else:
          status_label.value = f"<b>RMSE:</b> {current_rmse:.3f} ❌ KEEP TRAINING "
    
    # interactive output
    out = interactive_output(update_plot, {"w": w_slider, "b": b_slider})
    ui = VBox([HBox([w_slider, b_slider]), status_label])
    display(ui, out)

def interactive_gradient_descent(X_train, y_train):
    """
    Gradient Descent GUI (RMSE heatmap + model plot + RMSE vs iter)
    - Trains in standardized-x space (stable), displays in original space
    - Auto-run is FAST: no incremental plotting; redraws once at the end
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from ipywidgets import Button, FloatText, IntText, HBox, VBox, HTML, Checkbox, Output, ToggleButton
    from IPython.display import display, clear_output

    # ----- Data -----
    x1d = np.asarray(X_train).ravel()
    y1d = np.asarray(y_train).ravel()
    N = x1d.size

    # --- Standardization (feature only) ---
    mu_x   = float(np.mean(x1d))
    std_x  = float(np.std(x1d)) if np.std(x1d) > 0 else 1.0
    x_std  = (x1d - mu_x) / std_x

    # ----- Losses -----
    def mse(y_true, y_pred):
        return float(np.mean((y_true - y_pred)**2))

    def rmse(y_true, y_pred):
        return float(np.sqrt(mse(y_true, y_pred)))

    # ----- Param space mappings -----
    def to_orig(ws, bs):
        w = ws / std_x
        b = bs - (ws * mu_x) / std_x
        return float(w), float(b)

    def to_std(w, b):
        ws = w * std_x
        bs = b + w * mu_x
        return float(ws), float(bs)

    # ----- Models -----
    def yhat_wb_orig(w, b, x):
        return w * x + b

    def yhat_wsbs(ws, bs, xz=x_std):
        return ws * xz + bs

    # ----- Analytic gradients for MSE in standardized space -----
    def grad_mse_wsbs(ws, bs):
        yhat = yhat_wsbs(ws, bs, x_std)
        err  = yhat - y1d
        gw_s = (2.0 / N) * float(np.dot(err, x_std))
        gb_s = (2.0 / N) * float(np.sum(err))
        return gw_s, gb_s

    # ----- Heatmap grid (ORIGINAL parameter space), Z = RMSE -----
    w_ls, b_ls = np.polyfit(x1d, y1d, 1)
    w_span = 2.0 * max(1.0, abs(w_ls))
    b_span = 2.0 * max(1.0, abs(b_ls) if not np.isclose(b_ls, 0.0) else np.std(y1d))

    W = np.linspace(w_ls - w_span, w_ls + w_span, 150)
    B = np.linspace(b_ls - b_span, b_ls + b_span, 150)
    WW, BB = np.meshgrid(W, B)

    Z = np.zeros_like(WW, dtype=float)
    for i in range(WW.shape[0]):
        for j in range(WW.shape[1]):
            Z[i, j] = rmse(y1d, yhat_wb_orig(WW[i, j], BB[i, j], x1d))

    # ----- State -----
    ws, bs = to_std(1.0, 1.0)            # internal (standardized) params
    w_disp, b_disp = to_orig(ws, bs)     # display (original) params
    history_w = [w_disp]
    history_b = [b_disp]
    history_RMSE = [rmse(y1d, yhat_wb_orig(w_disp, b_disp, x1d))]

    # ----- Controls / Outputs -----
    alpha_init = 0.2
    alpha_box = FloatText(value=alpha_init, description='learning rate α', step=0.001)
    steps_box = IntText(value=1, description='steps/click')

    init_w_box = FloatText(value=1, description='init w (orig)')
    init_b_box = FloatText(value=1, description='init b (orig)')
    btn_apply  = Button(description='Apply Start', tooltip='Use init w/b (orig) without reset')

    auto_box   = Checkbox(value=False, description='Auto-run (until plateau or max iters)')
    max_box    = IntText(value=500, description='max iters (auto)')
    tol_box    = FloatText(value=1e-8, description='min ΔRMSE (auto)', step=1e-9)

    btn_step   = Button(description='Step', button_style='success')
    btn_reset  = Button(description='Reset', button_style='warning')
    pick_toggle= ToggleButton(value=False, description='Pick start on heatmap',
                              tooltip='Click on heatmap to set start (orig w,b)')

    status     = HTML(value='')
    out_plot   = Output()   # heatmap
    out_model  = Output()   # model vs data
    out_loss   = Output()   # RMSE vs iteration

    # refs for click picking
    fig_ref, ax_ref = None, None
    cid_click = None

    def _set_start_orig(new_w, new_b):
        nonlocal ws, bs, history_w, history_b, history_RMSE
        ws, bs = to_std(new_w, new_b)
        w_show, b_show = to_orig(ws, bs)
        history_w = [w_show]
        history_b = [b_show]
        history_RMSE = [rmse(y1d, yhat_wb_orig(w_show, b_show, x1d))]
        status.value = (
            f"<b>Start set (orig space):</b> w={w_show:.3f}, b={b_show:.3f}, "
            f"RMSE={history_RMSE[-1]:.6g}"
        )

    def draw_heatmap():
        nonlocal fig_ref, ax_ref, cid_click
        with out_plot:
            clear_output(wait=True)
            fig_ref = plt.figure(figsize=(6.4, 5.0))
            ax_ref = fig_ref.add_subplot(111)
            im = ax_ref.imshow(
                Z, origin='lower',
                extent=[W.min(), W.max(), B.min(), B.max()],
                aspect='auto',
                cmap='plasma'
            )
            ax_ref.plot(history_w, history_b, marker='o', linewidth=1.5)
            ax_ref.scatter([history_w[-1]], [history_b[-1]], label='current')
            ax_ref.scatter([w_ls], [b_ls], marker='x', label='Optimum')
            ax_ref.set_xlabel('w (slope, orig)')
            ax_ref.set_ylabel('b (intercept, orig)')
            ax_ref.set_title('Loss Heatmap (RMSE) with Optimization Path')
            fig_ref.colorbar(im, ax=ax_ref, label='RMSE L(w,b)')
            ax_ref.legend(loc='upper right')
            plt.show()

            if pick_toggle.value:
                _bind_click_handler()
            else:
                _unbind_click_handler()

    def draw_model_plot():
        with out_model:
            clear_output(wait=True)
            fig = plt.figure(figsize=(6.4, 5.0))
            ax = fig.add_subplot(111)

            ax.scatter(x1d, y1d, alpha=0.8, label='training data')
            w_show, b_show = to_orig(ws, bs)
            x_min, x_max = np.min(x1d), np.max(x1d)
            x_line = np.linspace(x_min, x_max, 200)
            ax.plot(x_line, yhat_wb_orig(w_show, b_show, x_line), linewidth=2.0,
                    label=f'current: y = {w_show:.3f}x + {b_show:.3f}')
            ax.plot(x_line, yhat_wb_orig(w_ls, b_ls, x_line), linestyle='--', linewidth=1.5,
                    label=f'Optimum: y = {w_ls:.3f}x + {b_ls:.3f}')
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_title('Current Model vs Training Data')
            ax.legend(loc='best')
            plt.show()

    def draw_loss_plot():
        with out_loss:
            clear_output(wait=True)
            fig = plt.figure(figsize=(6.4, 3.2))
            ax = fig.add_subplot(111)
            ax.plot(range(len(history_RMSE)), history_RMSE, marker='o', linewidth=1.5)
            ax.set_xlabel('Iteration'); ax.set_ylabel('RMSE')
            ax.set_title('Training Curve: RMSE vs Iteration')
            ax.grid(True, alpha=0.3)
            plt.show()

    def _on_click(event):
        if event.inaxes is not ax_ref:
            return
        new_w = float(np.clip(event.xdata, W.min(), W.max()))
        new_b = float(np.clip(event.ydata, B.min(), B.max()))
        _set_start_orig(new_w, new_b)
        _update_status()
        # Single redraw (still interactive for manual actions)
        draw_heatmap(); draw_model_plot(); draw_loss_plot()

    def _bind_click_handler():
        nonlocal cid_click
        if fig_ref is None:
            return
        if cid_click is None:
            cid_click = fig_ref.canvas.mpl_connect('button_press_event', _on_click)

    def _unbind_click_handler():
        nonlocal cid_click
        if fig_ref is None or cid_click is None:
            return
        fig_ref.canvas.mpl_disconnect(cid_click)
        cid_click = None

    def one_step():
        nonlocal ws, bs
        alpha = float(alpha_box.value)
        gw_s, gb_s = grad_mse_wsbs(ws, bs)
        ws = ws - alpha * gw_s
        bs = bs - alpha * gb_s

        w_show, b_show = to_orig(ws, bs)
        history_w.append(w_show)
        history_b.append(b_show)
        history_RMSE.append(rmse(y1d, yhat_wb_orig(w_show, b_show, x1d)))

    def _update_status():
        w_show, b_show = to_orig(ws, bs)
        status.value = (
            f"<b>Iter:</b> {len(history_RMSE)-1} "
            f"&nbsp;&nbsp; <b>w:</b> {w_show:.6g} "
            f"&nbsp;&nbsp; <b>b:</b> {b_show:.6g} "
            f"&nbsp;&nbsp; <b>RMSE:</b> {history_RMSE[-1]:.6g}"
        )

    def on_step_clicked(_):
        n = int(steps_box.value)
        for _ in range(max(n, 1)):
            one_step()
        _update_status()
        draw_heatmap(); draw_model_plot(); draw_loss_plot()
        # If auto-run is toggled, continue running (fast, no incremental plotting)
        if auto_box.value:
            auto_run()

    def on_reset_clicked(_):
        _set_start_orig(w_ls, b_ls)
        _update_status()
        draw_heatmap(); draw_model_plot(); draw_loss_plot()

    def on_apply_clicked(_):
        _set_start_orig(float(init_w_box.value), float(init_b_box.value))
        _update_status()
        draw_heatmap(); draw_model_plot(); draw_loss_plot()

    # ----- FAST Auto-run: no incremental plotting -----
    def auto_run():
        """Run GD until tol or max iters; plot once at the end."""
        it = 0
        max_iters = int(max_box.value)
        tol = float(tol_box.value)
        prev = history_RMSE[-1]

        # tight loop (no drawing)
        while auto_box.value and it < max_iters:
            one_step()
            it += 1
            curr = history_RMSE[-1]
            if abs(prev - curr) < tol:
                auto_box.value = False  # stop auto if converged
                break
            prev = curr

        # single redraw at end
        _update_status()
        draw_heatmap(); draw_model_plot(); draw_loss_plot()

    def on_auto_toggle(change):
        if change.get("name") == "value" and change["new"]:
            auto_run()  # start fast loop

    # Wire up events
    auto_box.observe(on_auto_toggle)
    pick_toggle.observe(lambda ch: (_bind_click_handler() if ch["new"] else _unbind_click_handler())
                        if ch.get("name") == "value" else None)
    btn_step.on_click(on_step_clicked)
    btn_reset.on_click(on_reset_clicked)
    btn_apply.on_click(on_apply_clicked)

    # ----- Initial render -----
    _set_start_orig(1.0, 1.0)
    _update_status()
    draw_heatmap(); draw_model_plot(); draw_loss_plot()

    controls_row1 = HBox([btn_step, steps_box, btn_reset])
    controls_row2 = HBox([alpha_box, init_w_box, init_b_box, btn_apply])
    controls_row3 = HBox([pick_toggle, auto_box, max_box, tol_box])

    top_row = HBox([out_plot, out_model])
    display(VBox([controls_row1, controls_row2, controls_row3, status, top_row, out_loss]))
    
import matplotlib.pyplot as plt
import numpy as np

def plot_performance(w, b, X_train, X_test, y_train, y_test):
    """Plot regression and parity plots side by side."""
    def model(x, w, b):
        return w*x + b

    # Predictions
    y_pred_train = model(X_train, w, b)
    y_pred_test = model(X_test, w, b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Regression Plot ---
    ax = axes[0]
    ax.scatter(X_train, y_train, label="Train data", color="blue", alpha=0.7)
    ax.scatter(X_test, y_test, label="Test data", color="green", alpha=0.7, marker="x")
    xs = np.linspace(min(X_train.min(), X_test.min()),
                     max(X_train.max(), X_test.max()), 200)
    ax.plot(xs, model(xs, w, b), color="red", label="Model prediction")
    ax.set_xlabel("Molecular Weight")
    ax.set_ylabel("Boiling Point (K)")
    ax.set_title("Regression Plot")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Parity Plot ---
    ax = axes[1]
    ax.scatter(y_train, y_pred_train, label="Train data", color="blue", alpha=0.7)
    ax.scatter(y_test, y_pred_test, label="Test data", color="green", alpha=0.7, marker="x")
    all_y = np.concatenate([y_train, y_test])
    ax.plot([all_y.min(), all_y.max()],
            [all_y.min(), all_y.max()],
            'k--', label="Perfect prediction")
    ax.set_xlabel("True Boiling Point (K)")
    ax.set_ylabel("Predicted Boiling Point (K)")
    ax.set_title("Parity Plot")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def interactive_polynomial_regression(X_train, X_test, y_train, y_test):
    """
    GUI: sklearn linear models with selectable x^k and log(x) features.
    - Internally STANDARDIZES features for fitting (always on)
    - Displays parameters mapped back to ORIGINAL feature space
    - Models: LinearRegression or Ridge (alpha)
    - Features: x, x^2, x^3, x^4, log(x) (log(x) ignored if any x <= 0)
    - Plots: fit curve vs data (left) + parity plot (right)
    - RETURNS a handle object with:
        .model, .scaler, .feature_names, .eqn,
        .predict(x_new), .transform(x_vec)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from ipywidgets import Checkbox, FloatText, Dropdown, Button, HBox, VBox, Output, HTML, Layout
    from IPython.display import display, clear_output
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score

    # -------- Data (expects 1D X) --------
    x_raw = np.asarray(X_train, dtype=float).ravel()
    y_raw = np.asarray(y_train, dtype=float).ravel()
    x_test_raw = np.asarray(X_test, dtype=float).ravel()   # <-- added
    y_test_raw = np.asarray(y_test, dtype=float).ravel()   # <-- added

    # -------- Widgets --------
    w_x   = Checkbox(value=True,  description="x")
    w_x2  = Checkbox(value=False, description="x²")
    w_x3  = Checkbox(value=False, description="x³")
    w_x4  = Checkbox(value=False, description="x⁴")
    w_log = Checkbox(value=False, description="log(x)")

    w_model = Dropdown(options=["LinearRegression", "Ridge"], value="LinearRegression", description="Model")
    w_alpha = FloatText(value=1.0, description="Lambda (Ridge)", step=0.1, disabled=True)

    btn_fit   = Button(description="Fit model", button_style="success")
    btn_reset = Button(description="Reset", button_style="warning")

    status   = HTML()
    out_fit  = Output()
    out_par  = Output()
    out_text = Output()

    # ------- handle object to return -------
    class _Handle:
        model = None
        scaler = None
        feature_names = None
        eqn = None
        def transform(self, x_vec):
            Xd, _, _ = _build_features(np.asarray(x_vec, dtype=float).ravel(), w_log.value)
            return Xd
        def predict(self, x_new):
            x_new = np.asarray(x_new, dtype=float).ravel()
            Xd, _, warn = _build_features(x_new, w_log.value)
            if warn and w_log.value:
                pass  # log(x) ignored for non-positive values; still proceed
            Xs = self.scaler.transform(Xd)
            return self.model.predict(Xs)

    handle = _Handle()

    # ------- feature builder -------
    def _build_features(x, use_log):
        """
        Build design matrix from selected widgets.
        Returns X_design, feature_names, warning_html ('')
        """
        feats, names = [], []
        warn = ""
        if w_x.value:
            feats.append(x); names.append("x")
        if w_x2.value:
            feats.append(x**2); names.append("x^2")
        if w_x3.value:
            feats.append(x**3); names.append("x^3")
        if w_x4.value:
            feats.append(x**4); names.append("x^4")
        if use_log:
            if np.any(x <= 0):
                warn = "<i>log(x) ignored because some x ≤ 0.</i>"
            else:
                feats.append(np.log(x)); names.append("log(x)")
        if len(feats) == 0:
            feats = [x]; names = ["x"]
        X_design = np.column_stack(feats)
        return X_design, names, warn

    # ------- main fit/plot routine -------
    def _fit_and_plot(*_):
        # Build TRAIN features
        X_design, names, warn = _build_features(x_raw, w_log.value)

        # Always standardize for fitting
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_fit = scaler.fit_transform(X_design)

        # Model selection
        if w_model.value == "LinearRegression":
            model = LinearRegression(fit_intercept=True)
        else:
            model = Ridge(alpha=float(w_alpha.value), fit_intercept=True, random_state=0)

        # Fit & predict (train)
        model.fit(X_fit, y_raw)
        y_pred = model.predict(X_fit)

        # --- TEST predictions & metrics (added) ---
        X_test_design, _, warn_test = _build_features(x_test_raw, w_log.value)
        X_test_fit = scaler.transform(X_test_design)
        y_pred_test = model.predict(X_test_fit)
        rmse_test = float(np.sqrt(mean_squared_error(y_test_raw, y_pred_test)))
        # ------------------------------------------

        # Metrics (train)
        rmse = float(np.sqrt(mean_squared_error(y_raw, y_pred)))
        r2   = float(r2_score(y_raw, y_pred))

        # Smooth curve spanning train+test x-range
        x_min = float(np.min(np.hstack((x_raw, x_test_raw))))
        x_max = float(np.max(np.hstack((x_raw, x_test_raw))))
        x_line = np.linspace(x_min, x_max, 400)
        X_line_design, _, warn_line = _build_features(x_line, w_log.value)
        X_line_fit = scaler.transform(X_line_design)
        y_line_pred = model.predict(X_line_fit)

        # Map params back to ORIGINAL feature space
        coef_std = np.asarray(model.coef_).ravel()
        intercept_std = float(model.intercept_)
        mu = scaler.mean_.ravel()
        sigma = scaler.scale_.ravel()
        sigma_safe = np.where(sigma == 0, 1.0, sigma)
        coef_orig = coef_std / sigma_safe
        intercept_orig = intercept_std - np.sum(coef_std * (mu / sigma_safe))

        # Human-readable eqn (original space)
        terms = [f"{coef_orig[i]:+.4g}·{names[i]}" for i in range(len(names))]
        eqn = f"y = {intercept_orig:+.4g} " + " ".join(terms) if terms else f"y = {intercept_orig:+.4g}"

        # Update handle
        handle.model = model
        handle.scaler = scaler
        handle.feature_names = names
        handle.eqn = eqn

        # Status
        extra_warn = warn or warn_line or warn_test
        status.value = (
            f"<b>Model:</b> {w_model.value} &nbsp;&nbsp; "
            f"<b>Features:</b> {', '.join(names)}<br>"
            f"<b>RMSE (train):</b> {rmse:.5g} &nbsp;&nbsp; "
            f"<b>RMSE (test):</b> {rmse_test:.5g}"
            f"{' • ' + extra_warn if extra_warn else ''}"
        )

        # Left plot: fit curve vs data
        with out_fit:
            clear_output(wait=True)
            fig = plt.figure(figsize=(7.0, 5.0))
            ax = fig.add_subplot(111)
            ax.scatter(x_raw, y_raw, alpha=0.85, label="training data")
            ax.scatter(x_test_raw, y_test_raw, alpha=0.85, label="testing data")
            ax.plot(x_line, y_line_pred, linewidth=2.0, label="model prediction")
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_title("Model fit with selected features")
            ax.legend(loc="best")
            plt.show()

        # Parity plot (fixed lo/hi computation)
        with out_par:
            clear_output(wait=True)
            fig = plt.figure(figsize=(7.0, 5.0))
            ax = fig.add_subplot(111)

            ax.scatter(y_raw,      y_pred,      alpha=0.85, label="train")
            ax.scatter(y_test_raw, y_pred_test, alpha=0.85, label="test")

            lo = float(np.min([y_raw.min(), y_pred.min(), y_test_raw.min(), y_pred_test.min()]))
            hi = float(np.max([y_raw.max(), y_pred.max(), y_test_raw.max(), y_pred_test.max()]))

            ax.plot([lo, hi], [lo, hi], linewidth=1.5, linestyle="--", label="y = x")
            ax.set_xlabel("Actual y"); ax.set_ylabel("Predicted ŷ")
            ax.set_title("Parity plot (ŷ vs y)")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            plt.show()


        # Equation / coefficients box
        with out_text:
            clear_output(wait=True)
            print("Learned equation (ORIGINAL feature space):")
            print("  ", eqn)
            print()
            print("Coefficients (original features):")
            for name, c in zip(names, coef_orig):
                print(f"  {name:>10s}: {c: .6g}")
            print(f"  {'intercept':>10s}: {intercept_orig: .6g}")

    def _reset(*_):
        w_x.value, w_x2.value, w_x3.value, w_x4.value, w_log.value = True, False, False, False, False
        w_model.value = "LinearRegression"
        w_alpha.value = 1.0
        _on_model_change({"new": w_model.value})
        _fit_and_plot()

    def _on_model_change(change):
        if change.get("new", w_model.value) == "Ridge":
            w_alpha.disabled = False
        else:
            w_alpha.disabled = True

    # Wire up + layout
    btn_fit.on_click(_fit_and_plot)
    btn_reset.on_click(_reset)
    w_model.observe(_on_model_change, names="value")

    controls_row1 = HBox([w_x, w_x2, w_x3, w_x4, w_log], layout=Layout(gap="12px"))
    controls_row2 = HBox([w_model, w_alpha, btn_fit, btn_reset], layout=Layout(gap="12px"))

    two_plots = HBox([out_fit, out_par], layout=Layout(gap="24px"))
    display(VBox([controls_row1, controls_row2, status, two_plots, out_text]))

    # initial fit (so handle is populated immediately)
    _fit_and_plot()

    return handle
    
import matplotlib.pyplot as plt
import numpy as np

def plot_model_losses(model_names, train_losses, test_losses):
    """
    Plot training and testing RMSE for multiple models as a grouped bar chart.

    Args:
        model_names (list of str): names of models (e.g., ["Model A", "Model B"])
        train_losses (list of float): training RMSE for each model
        test_losses (list of float): testing RMSE for each model
    """
    x = np.arange(len(model_names))  # positions
    width = 0.35                     # bar width

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_train = ax.bar(x - width/2, train_losses, width,
                        label="Train", color="steelblue", alpha=0.85)
    bars_test = ax.bar(x + width/2, test_losses, width,
                       label="Test", color="seagreen", alpha=0.85)

    # Add labels inside bars
    for bars in [bars_train, bars_test]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f"{height:.2f}", ha='center', va='center',
                    fontsize=10, color="white", fontweight="bold")

    ax.set_ylabel("RMSE Loss")
    ax.set_title("Model Performance: Training vs Testing Loss")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_multiple_linear_regression(X_train, X_test, y_train, y_test, renderer: str | None = None):
    """
    Colab-friendly 3D regression plot for:
        features = [log(MW), branching_index]  ->  bp_k
    Returns the Plotly figure (does not auto-plot).
    """
    import numpy as np, pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    import plotly.graph_objects as go, plotly.io as pio

    # Renderer: default to 'colab'
    if renderer is None:
        try:
            pio.renderers.default = "colab"
            renderer_used = "colab"
        except Exception:
            pio.renderers.default = "png"
            renderer_used = "png"
    else:
        pio.renderers.default = renderer
        renderer_used = renderer

    # Safety checks
    for name, arr in [("X_train", X_train), ("X_test", X_test),
                      ("y_train", y_train), ("y_test", y_test)]:
        if arr is None:
            raise ValueError(f"{name} is not defined")
    for col in ["MW", "branching_index"]:
        if col not in X_train or col not in X_test:
            raise ValueError(f"Missing {col} in dataset")
    if not (np.all(X_train["MW"] > 0) and np.all(X_test["MW"] > 0)):
        raise ValueError("All MW values must be > 0")

    # Build features
    lmw_tr, bi_tr = np.log(X_train["MW"].values), X_train["branching_index"].values
    lmw_te, bi_te = np.log(X_test["MW"].values),  X_test["branching_index"].values
    Xtr, Xte = np.column_stack([lmw_tr, bi_tr]), np.column_stack([lmw_te, bi_te])
    ytr, yte = np.asarray(y_train).ravel(), np.asarray(y_test).ravel()

    # Fit model
    model = LinearRegression().fit(Xtr, ytr)
    ytr_pred, yte_pred = model.predict(Xtr), model.predict(Xte)

    # Metrics
    rmse_tr = float(np.sqrt(mean_squared_error(ytr, ytr_pred)))
    rmse_te = float(np.sqrt(mean_squared_error(yte, yte_pred)))
    r2_tr, r2_te = r2_score(ytr, ytr_pred), r2_score(yte, yte_pred)

    # Plane
    lmw_grid = np.linspace(min(lmw_tr.min(), lmw_te.min()),
                           max(lmw_tr.max(), lmw_te.max()), 40)
    bi_grid  = np.linspace(min(bi_tr.min(), bi_te.min()),
                           max(bi_tr.max(), bi_te.max()), 40)
    LMW, BI = np.meshgrid(lmw_grid, bi_grid)
    plane_Y = model.predict(np.column_stack([LMW.ravel(), BI.ravel()])).reshape(LMW.shape)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=lmw_tr, y=bi_tr, z=ytr,
                               mode="markers", name="train",
                               marker=dict(size=4, color="royalblue")))
    fig.add_trace(go.Scatter3d(x=lmw_te, y=bi_te, z=yte,
                               mode="markers", name="test",
                               marker=dict(size=4, color="firebrick")))
    fig.add_trace(go.Surface(x=LMW, y=BI, z=plane_Y,
                             name="fit plane", showscale=False, opacity=0.5))

    fig.update_layout(
        width=900, height=700,
        title="Linear Regression Plane • log(MW), branching_index → bp_k",
        scene=dict(
            xaxis_title="log(MW)", yaxis_title="branching_index", zaxis_title="Boiling Point (K)",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.4), projection=dict(type="orthographic"))
        )
    )

    # Print info
    coef, intercept = model.coef_.ravel(), float(model.intercept_)
    print("Fitted equation:")
    print(f"  y = {intercept:.6g} + ({coef[0]:.6g})·log(MW) + ({coef[1]:.6g})·branching_index")
    print("\nMetrics:")
    print(f"  Train RMSE = {rmse_tr:.6g}  Train R² = {r2_tr:.6g}")
    print(f"  Test  RMSE = {rmse_te:.6g}  Test  R² = {r2_te:.6g}")
    print(f"[Renderer: {renderer_used}]")

    return fig

def interactive_multiple_linear_regression(X_train, X_test, y_train, y_test):
    """
    GUI for predicting boiling point (bp_k) using selectable features:
      - MW
      - log(MW)   (ignored if any MW <= 0 in either split; a note is shown)
      - branching_index

    Behavior:
      - If exactly one feature is selected -> fitted line vs that feature (plus actual train/test points).
      - If multiple features are selected -> plot predictions as points vs MW (ŷ vs MW) for TRAIN and TEST,
        with faint gray actual points for context.

    Requires: X_train, X_test (pandas DataFrames), y_train, y_test (array-like).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from ipywidgets import Checkbox, HBox, VBox, Output, HTML, Button
    from IPython.display import display, clear_output
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # ---- Safety checks ----
    for name, arr in [("X_train", X_train), ("X_test", X_test),
                      ("y_train", y_train), ("y_test", y_test)]:
        if arr is None:
            raise ValueError(f"{name} is not defined")
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_train and X_test must be pandas DataFrames.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("X/y train/test lengths do not match.")

    def _ensure_cols(X, cols):
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"X is missing required columns: {missing}")

    # ---- Widgets ----
    w_mw   = Checkbox(value=True,  description="MW")
    w_lmw  = Checkbox(value=False, description="log(MW)")
    w_bi   = Checkbox(value=False, description="branching_index")
    btn_fit = Button(description="Fit / Plot", button_style="success")
    status  = HTML()
    out     = Output()

    # ---- Helpers ----
    def _build_selected_names():
        """Return list of selected feature names validated across both splits."""
        selected = []
        if w_mw.value:
            selected.append("MW")
        if w_lmw.value and ("MW" in X_train.columns) and ("MW" in X_test.columns):
            mw_tr = X_train["MW"].values
            mw_te = X_test["MW"].values
            if (mw_tr > 0).all() and (mw_te > 0).all():
                selected.append("log(MW)")
        if w_bi.value:
            selected.append("branching_index")
        if len(selected) == 0:
            selected = ["MW"]
        return selected

    def _design_from_names(names, X_df):
        cols = []
        for nm in names:
            if nm == "MW":
                _ensure_cols(X_df, ["MW"])
                cols.append(X_df["MW"].values)
            elif nm == "log(MW)":
                _ensure_cols(X_df, ["MW"])
                cols.append(np.log(X_df["MW"].values))
            elif nm == "branching_index":
                _ensure_cols(X_df, ["branching_index"])
                cols.append(X_df["branching_index"].values)
        return np.column_stack(cols)

    def _fit_and_plot(_=None):
        with out:
            clear_output(wait=True)

            # Active features and design matrices
            active = _build_selected_names()
            Xtr = _design_from_names(active, X_train)
            Xte = _design_from_names(active, X_test)
            ytr = np.asarray(y_train).ravel()
            yte = np.asarray(y_test).ravel()

            model = LinearRegression()
            model.fit(Xtr, ytr)

            ytr_pred = model.predict(Xtr)
            yte_pred = model.predict(Xte)

            rmse_tr = float(np.sqrt(mean_squared_error(ytr, ytr_pred)))
            r2_tr   = float(r2_score(ytr, ytr_pred))
            rmse_te = float(np.sqrt(mean_squared_error(yte, yte_pred)))
            r2_te   = float(r2_score(yte, yte_pred))

            mw_tr = X_train["MW"].values if "MW" in X_train.columns else None
            mw_te = X_test["MW"].values  if "MW" in X_test.columns  else None

            plt.figure(figsize=(8.6, 5.6))

            if len(active) == 1:
                # --- Single feature: fitted line + actuals ---
                feat = active[0]
                if feat == "MW":
                    xtr = mw_tr; xte = mw_te; xlabel = "MW"
                    x_all = np.concatenate([xtr, xte])
                    xg = np.linspace(x_all.min(), x_all.max(), 200).reshape(-1, 1)
                    yg = model.predict(xg)

                    plt.scatter(xtr, ytr, label="train actual", alpha=0.9)
                    plt.scatter(xte, yte, label="test actual",  alpha=0.9)
                    plt.plot(xg.ravel(), yg, color="black", linewidth=2.2, label="fit (train)")

                elif feat == "log(MW)":
                    xtr = np.log(mw_tr); xte = np.log(mw_te); xlabel = "log(MW)"
                    x_all = np.concatenate([xtr, xte])
                    xg = np.linspace(x_all.min(), x_all.max(), 200).reshape(-1, 1)
                    yg = model.predict(xg)

                    plt.scatter(xtr, ytr, label="train actual", alpha=0.9)
                    plt.scatter(xte, yte, label="test actual",  alpha=0.9)
                    plt.plot(xg.ravel(), yg, color="black", linewidth=2.2, label="fit (train)")

                else:  # branching_index
                    xtr = X_train["branching_index"].values
                    xte = X_test["branching_index"].values
                    xlabel = "branching_index"
                    x_all = np.concatenate([xtr, xte])
                    xg = np.linspace(x_all.min(), x_all.max(), 200).reshape(-1, 1)
                    yg = model.predict(xg)

                    plt.scatter(xtr, ytr, label="train actual", alpha=0.9)
                    plt.scatter(xte, yte, label="test actual",  alpha=0.9)
                    plt.plot(xg.ravel(), yg, color="black", linewidth=2.2, label="fit (train)")

                title = f"Linear Regression • {feat}"

            else:
                # --- Multi-feature: predictions vs MW (plus faint actuals for context) ---
                if ("MW" not in X_train.columns) or ("MW" not in X_test.columns):
                    raise ValueError("MW column is required to plot predictions vs MW for multi-feature models.")

                # Context: faint actuals
                plt.scatter(mw_tr, ytr, label="train actual (faint)", alpha=0.25, s=30, color="gray", zorder=1)
                plt.scatter(mw_te, yte, label="test actual (faint)",  alpha=0.25, s=30, color="lightgray", zorder=1)

                # Preds: hollow, colored edges, high z-order so they pop
                plt.scatter(mw_tr, ytr_pred, label="ŷ train (vs MW)", alpha=1.0, s=70,
                            facecolors='none', edgecolors='tab:orange', linewidths=2.0, zorder=5)
                plt.scatter(mw_te, yte_pred, label="ŷ test (vs MW)",  alpha=1.0, s=90, marker='^',
                            facecolors='none', edgecolors='tab:green', linewidths=2.2, zorder=6)

                xlabel = "MW"
                title = f"Linear Regression • {', '.join(active)} (predictions vs MW)"

            plt.xlabel(xlabel)
            plt.ylabel("Boiling Point (K)")
            plt.title(title)
            plt.legend()
            plt.show()

            # Equation + metrics
            coef = model.coef_.ravel()
            intercept = float(model.intercept_)
            print("Equation (features in order):", ", ".join(active))
            if len(active) == 1:
                print(f"  y = {intercept:.6g} + ({coef[0]:.6g})·{active[0]}")
            else:
                terms = " + ".join([f"({coef[i]:.6g})·{active[i]}" for i in range(len(active))])
                print(f"  y = {intercept:.6g} + {terms}")

            # Warnings
            warn_msgs = []
            if w_lmw.value and "log(MW)" not in active:
                warn_msgs.append("log(MW) ignored because some MW ≤ 0 in train or test.")
            status.value = ("<b>Note:</b> " + " • ".join(warn_msgs)) if warn_msgs else ""

            print("\nMetrics:")
            print(f"  Train RMSE = {rmse_tr:.6g}    Train R² = {r2_tr:.6g}")
            print(f"  Test  RMSE = {rmse_te:.6g}    Test  R² = {r2_te:.6g}")

    # Wire up + display
    controls = HBox([w_mw, w_lmw, w_bi, btn_fit])
    display(VBox([controls, status, out]))
    btn_fit.on_click(_fit_and_plot)

    # Initial fit
    _fit_and_plot()


